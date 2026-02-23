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

from tqdm import tqdm
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
from langchain_core.documents import Document


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
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _should_exclude(self, year: Optional[int]) -> bool:
        """Check if document should be excluded based on year."""
        if self.exclude_year_gte is None or year is None:
            return False
        return year >= self.exclude_year_gte

    # -------------------------------------------------------------------------
    # Single File Cache Operations (for BERT/RAG pipelines)
    # -------------------------------------------------------------------------

    def get_cache_path(self, pdf_stem: str, suffix: str) -> Path:
        """Get cache file path for a specific PDF.

        Args:
            pdf_stem: PDF filename without extension (e.g., "report_2018")
            suffix: Cache type suffix ("prep" or "bert")

        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{pdf_stem}_{suffix}.json"

    def load_single_cache(self, pdf_stem: str, suffix: str) -> Optional[Dict]:
        """Load a single cache file by PDF stem and suffix.

        Args:
            pdf_stem: PDF filename without extension
            suffix: Cache type suffix ("prep" or "bert")

        Returns:
            Loaded JSON data or None if not found
        """
        cache_path = self.get_cache_path(pdf_stem, suffix)
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def save_single_cache(self, pdf_stem: str, suffix: str, data: Dict):
        """Save data to a single cache file.

        Args:
            pdf_stem: PDF filename without extension
            suffix: Cache type suffix ("prep" or "bert")
            data: Data to save
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.get_cache_path(pdf_stem, suffix)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

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


"""
Data Loader Module
==================

Shared data loading utilities for ML pipelines.
Supports loading both JSON and CSV files.

Usage:
    from data_loader import DataLoader, load_csv_data, load_json_data
"""


class DataLoader:
    """
    Unified data loader for JSON and CSV files.

    Usage:
        loader = DataLoader()

        # Load CSV files
        df = loader.load_csv_folder('data/', prefix='barriers', text_column='barriers')

        # Load JSON files
        data = loader.load_json_folder('data/', pattern='*.json')

        # Load single files
        df = loader.load_csv('file.csv')
        data = loader.load_json('file.json')
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize DataLoader.

        Args:
            verbose: Whether to print progress information
        """
        self.verbose = verbose

    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    # ==================== CSV Loading ====================

    def load_csv(
        self,
        filepath: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """
        Load a single CSV file.

        Args:
            filepath: Path to CSV file
            **kwargs: Additional arguments passed to pd.read_csv

        Returns:
            DataFrame with loaded data
        """
        return pd.read_csv(filepath, **kwargs)

    def load_csv_folder(
        self,
        folder_path: Union[str, Path],
        prefix: Optional[str] = None,
        suffix: str = '.csv',
        text_column: Optional[str] = None,
        filter_pattern: Optional[str] = 'NO_',
        **kwargs
    ) -> pd.DataFrame:
        """
        Load multiple CSV files from a folder.

        Args:
            folder_path: Path to folder containing CSV files
            prefix: Only load files starting with this prefix (e.g., 'barriers')
            suffix: File extension to look for (default: '.csv')
            text_column: Column to filter on (removes rows containing filter_pattern)
            filter_pattern: Pattern to filter out from text_column (e.g., 'NO_')
            **kwargs: Additional arguments passed to pd.read_csv

        Returns:
            Concatenated DataFrame from all matching files
        """
        folder = Path(folder_path)

        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Find matching files
        files = [
            f for f in folder.iterdir()
            if f.suffix.lower() == suffix.lower()
            and (prefix is None or f.name.startswith(prefix))
        ]

        if not files:
            self._log(f"⚠️  No {suffix} files found in {folder_path}" +
                      (f" with prefix '{prefix}'" if prefix else ""))
            return pd.DataFrame()

        self._log(f"\n📂 Loading {len(files)} CSV files from {folder_path}...")

        all_data = []
        iterator = tqdm(
            files, desc="Reading CSV files") if self.verbose else files

        for filepath in iterator:
            try:
                df = pd.read_csv(filepath, **kwargs)

                # Filter out unwanted rows if specified
                if text_column and filter_pattern and text_column in df.columns:
                    df = df[~df[text_column].str.contains(
                        filter_pattern, na=False)]

                all_data.append(df)
            except Exception as e:
                self._log(f"⚠️  Error loading {filepath.name}: {e}")

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        self._log(f"✅ Loaded {len(result)} rows from {len(all_data)} files")

        return result

    # ==================== JSON Loading ====================

    def load_json(
        self,
        filepath: Union[str, Path]
    ) -> Union[Dict, List]:
        """
        Load a single JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Parsed JSON data (dict or list)
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_json_folder(
        self,
        folder_path: Union[str, Path],
        prefix: Optional[str] = None,
        suffix: str = '.json',
        flatten: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Load multiple JSON files from a folder.

        Args:
            folder_path: Path to folder containing JSON files
            prefix: Only load files starting with this prefix
            suffix: File extension to look for (default: '.json')
            flatten: If True and JSON files contain lists, flatten into single list

        Returns:
            List of loaded JSON data with source file information
        """
        folder = Path(folder_path)

        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Find matching files
        files = [
            f for f in folder.iterdir()
            if f.suffix.lower() == suffix.lower()
            and (prefix is None or f.name.startswith(prefix))
        ]

        if not files:
            self._log(f"⚠️  No {suffix} files found in {folder_path}" +
                      (f" with prefix '{prefix}'" if prefix else ""))
            return []

        self._log(f"\n📂 Loading {len(files)} JSON files from {folder_path}...")

        all_data = []
        iterator = tqdm(
            files, desc="Reading JSON files") if self.verbose else files

        for filepath in iterator:
            try:
                data = self.load_json(filepath)

                if flatten and isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            item['_source_file'] = filepath.name
                        all_data.append(item)
                else:
                    if isinstance(data, dict):
                        data['_source_file'] = filepath.name
                    all_data.append(data)

            except Exception as e:
                self._log(f"⚠️  Error loading {filepath.name}: {e}")

        self._log(f"✅ Loaded {len(all_data)} items from {len(files)} files")
        return all_data

    def load_json_to_dataframe(
        self,
        folder_path: Union[str, Path],
        prefix: Optional[str] = None,
        record_path: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load JSON files and convert to DataFrame.

        Args:
            folder_path: Path to folder containing JSON files
            prefix: Only load files starting with this prefix
            record_path: Path to records in JSON (for nested structures)
            **kwargs: Additional arguments passed to pd.json_normalize

        Returns:
            DataFrame with normalized JSON data
        """
        data = self.load_json_folder(folder_path, prefix=prefix, flatten=True)

        if not data:
            return pd.DataFrame()

        return pd.json_normalize(data, record_path=record_path, **kwargs)


# ==================== Convenience Functions ====================

def load_csv_data(
    folder_path: str,
    file_type: str = 'barriers',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load CSV data (backwards compatible with original notebook function).

    Args:
        folder_path: Path to folder containing CSV files
        file_type: Type of data to load ('barriers' or 'motivators')
        verbose: Whether to print progress

    Returns:
        DataFrame with loaded and filtered data
    """
    loader = DataLoader(verbose=verbose)
    return loader.load_csv_folder(
        folder_path,
        prefix=file_type,
        text_column=file_type,
        filter_pattern='NO_'
    )


def load_json_data(
    folder_path: str,
    prefix: Optional[str] = None,
    flatten: bool = True,
    verbose: bool = True
) -> List[Dict]:
    """
    Load JSON data from a folder.

    Args:
        folder_path: Path to folder containing JSON files
        prefix: Only load files starting with this prefix
        flatten: If True, flatten nested lists
        verbose: Whether to print progress

    Returns:
        List of loaded JSON data
    """
    loader = DataLoader(verbose=verbose)
    return loader.load_json_folder(folder_path, prefix=prefix, flatten=flatten)


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
    print()
    print("  # Single file operations")
    print("  loader = CacheLoader('cache')")
    print("  data = loader.load_single_cache('report_2018', 'prep')")
    print("  loader.save_single_cache('report_2018', 'bert', results)")

    # Example usage
    print("Data Loader Module")
    print("==================")
    print("\nUsage examples:")
    print("  from data_loader import DataLoader, load_csv_data")
    print("  ")
    print("  # Load barriers/motivators CSV files")
    print("  barriers_df = load_csv_data('data/', 'barriers')")
    print("  ")
    print("  # Use DataLoader class for more control")
    print("  loader = DataLoader()")
    print("  df = loader.load_csv_folder('data/', prefix='barriers')")
    print("  json_data = loader.load_json_folder('data/', prefix='config')")
