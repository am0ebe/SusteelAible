"""
PDF Preprocessing Module
========================

Shared preprocessing utilities for PDF extraction, cleaning, and chunking.
Used by both RAG and BERT analysis pipelines.

Usage:
    from pdf_preprocessing import PDFPreprocessor, PreprocessingConfig

    preprocessor = PDFPreprocessor()
    documents = preprocessor.process_folder("../data/reports")

    # Or single file
    doc = preprocessor.process_pdf("path/to/file.pdf")
"""

import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from collections import Counter

import fitz  # PyMuPDF
import spacy
import langid
from tqdm import tqdm

# Load spacy once at module level
_nlp = None


def _get_nlp():
    """Lazy-load spacy model."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=[
                          "ner", "lemmatizer", "tagger"])
        _nlp.max_length = 2_000_000
    return _nlp


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PreprocessingConfig:
    """Configuration for PDF preprocessing."""

    # Chunking parameters
    min_chunk_chars: int = 600
    max_chunk_chars: int = 1600

    # For LangChain-style chunking (RAG pipeline)
    langchain_chunk_size: int = 1500
    langchain_chunk_overlap: int = 100

    # Noise filtering
    min_paragraph_chars: int = 30

    # Characters considered as spam/noise
    spam_chars: str = "•·*─―–-"

    # Whether to detect and skip table content
    skip_tables: bool = True
    table_overlap_threshold: float = 0.5

    # Language detection
    detect_language: bool = True


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ProcessedDocument:
    """Represents a processed PDF document."""

    # Source info
    source_path: str
    filename: str

    # Extracted metadata
    company_name: Optional[str] = None
    company_id: Optional[str] = None
    year: Optional[int] = None
    report_type: Optional[str] = None

    # Content
    raw_text: str = ""
    cleaned_text: str = ""
    chunks: List[str] = field(default_factory=list)

    # Language
    language: str = "en"

    # Processing info
    num_pages: int = 0
    extraction_method: str = "pymupdf"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "source_path": self.source_path,
            "filename": self.filename,
            "company_name": self.company_name,
            "company_id": self.company_id,
            "year": self.year,
            "report_type": self.report_type,
            "language": self.language,
            "num_pages": self.num_pages,
            "num_chunks": len(self.chunks),
            "extraction_method": self.extraction_method,
        }


@dataclass
class DocumentChunk:
    """Represents a single chunk from a document."""

    chunk_id: str
    text: str

    # Parent document info
    source_path: str
    filename: str
    company_name: Optional[str] = None
    company_id: Optional[str] = None
    year: Optional[str] = None
    report_type: Optional[str] = None
    language: str = "en"

    # Position info
    chunk_index: int = 0

    def to_langchain_document(self):
        """Convert to LangChain Document format."""
        from langchain.schema import Document
        return Document(
            page_content=self.text,
            metadata={
                "chunk_id": self.chunk_id,
                "source_path": self.source_path,
                "filename": self.filename,
                "company_name": self.company_name,
                "company_id": self.company_id,
                "year": self.year,
                "report_type": self.report_type,
                "language": self.language,
                "chunk_index": self.chunk_index,
            }
        )


# =============================================================================
# MAIN PREPROCESSOR CLASS
# =============================================================================

class PDFPreprocessor:
    """
    PDF preprocessing for climate report analysis.

    Handles:
    - PDF text extraction (with table detection)
    - Text cleaning and noise removal
    - Intelligent chunking
    - Metadata extraction from file paths
    - Language detection

    Usage:
        preprocessor = PDFPreprocessor()

        # Process single PDF
        doc = preprocessor.process_pdf("report.pdf")

        # Process folder
        docs = preprocessor.process_folder("reports/")

        # Get LangChain-compatible chunks
        chunks = preprocessor.get_langchain_chunks(docs)
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize preprocessor with optional custom config."""
        self.config = config or PreprocessingConfig()
        self._spam_chars_set = set(self.config.spam_chars)

    # -------------------------------------------------------------------------
    # PDF Text Extraction
    # -------------------------------------------------------------------------

    def extract_text_from_pdf(self, pdf_path: Union[str, Path]) -> Tuple[str, int]:
        """
        Extract text from PDF using PyMuPDF with table detection.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (extracted_text, num_pages)
        """
        pdf_path = Path(pdf_path)
        all_text = []
        num_pages = 0

        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
        except Exception as e:
            print(f"⚠️ Failed to open PDF {pdf_path}: {e}")
            return "", 0

        for page in doc:
            # Detect table bounding boxes
            table_bboxes = []
            if self.config.skip_tables:
                try:
                    tables = page.find_tables()
                    if tables:
                        table_bboxes = [table.bbox for table in tables]
                except:
                    pass

            # Try structured extraction first
            try:
                blocks = page.get_text("dict", sort=True)["blocks"]
            except:
                # Fallback to simple text extraction
                text = page.get_text("text")
                if text:
                    all_text.append(text)
                continue

            page_paragraphs = []
            for block in blocks:
                # Skip non-text blocks
                if block.get("type") != 0:
                    continue

                bbox = block.get("bbox", [0, 0, 0, 0])
                block_height = bbox[3] - bbox[1]

                # Skip very small blocks
                if block_height < 5:
                    continue

                # Skip content inside tables
                if self.config.skip_tables:
                    is_in_table = any(
                        self._bbox_overlap(bbox, tb)
                        for tb in table_bboxes
                    )
                    if is_in_table:
                        continue

                # Extract text from block
                block_lines = []
                for line in block.get("lines", []):
                    spans_text = [
                        span.get("text", "").strip()
                        for span in line.get("spans", [])
                        if span.get("text", "").strip()
                    ]
                    if spans_text:
                        block_lines.append(" ".join(spans_text))

                if block_lines:
                    paragraph = " ".join(block_lines)
                    paragraph = re.sub(r'\s+', ' ', paragraph).strip()
                    if paragraph and len(paragraph) > 2:
                        page_paragraphs.append(paragraph)

            if page_paragraphs:
                all_text.append("\n\n".join(page_paragraphs))

        doc.close()
        return "\n\n".join(all_text), num_pages

    def _bbox_overlap(self, bbox1, bbox2) -> bool:
        """Check if two bounding boxes overlap significantly."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

        if x_overlap == 0 or y_overlap == 0:
            return False

        intersection_area = x_overlap * y_overlap
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)

        if bbox1_area <= 0:
            return False

        return (intersection_area / bbox1_area) > self.config.table_overlap_threshold

    # -------------------------------------------------------------------------
    # Text Cleaning
    # -------------------------------------------------------------------------

    def clean_text(self, raw_text: str) -> str:
        """
        Clean and preprocess extracted text.

        Steps:
        1. Fix encoding issues
        2. Remove noise lines
        3. Clean artifacts
        4. Remove repetitions
        5. Reconstruct paragraphs

        Args:
            raw_text: Raw extracted text

        Returns:
            Cleaned text
        """
        # Fix encoding
        text = self._fix_encoding(raw_text)

        # Fix hyphenation at line breaks
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

        # Process lines
        lines = []
        for line in text.split('\n'):
            line = re.sub(r'[ \t]+', ' ', line).strip()
            if not line:
                lines.append('')
            elif self._is_noise_line(line):
                continue
            else:
                lines.append(line)

        # Reconstruct paragraphs
        paragraphs = []
        current = []

        for i, line in enumerate(lines):
            if line:
                current.append(line)
                ends_sentence = re.search(r'[.!?]\s*$', line)
                if ends_sentence and (i + 1 >= len(lines) or not lines[i + 1]):
                    para = ' '.join(current)
                    para = self._clean_artifacts(para)
                    if len(para) > self.config.min_paragraph_chars:
                        if not self._detect_severe_repetition(para):
                            paragraphs.append(para)
                    current = []
            elif current:
                if current and re.search(r'[.!?]\s*$', current[-1]):
                    para = ' '.join(current)
                    para = self._clean_artifacts(para)
                    if len(para) > self.config.min_paragraph_chars:
                        if not self._detect_severe_repetition(para):
                            paragraphs.append(para)
                    current = []

        # Handle remaining content
        if current:
            para = ' '.join(current)
            para = self._clean_artifacts(para)
            if len(para) > self.config.min_paragraph_chars:
                if not self._detect_severe_repetition(para):
                    paragraphs.append(para)

        return '\n\n'.join(paragraphs)

    def _fix_encoding(self, text: str) -> str:
        """Fix common PDF encoding issues."""
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'\(cid:\d+\)', '', text)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

        # Remove zero-width and invisible characters
        for char in ['\u00ad', '\u200b', '\u200c', '\u200d', '\ufeff']:
            text = text.replace(char, '')

        return text

    def _is_noise_line(self, line: str) -> bool:
        """Check if a line is noise (headers, footers, page numbers, etc.)."""
        line = line.strip()

        # Too short
        if len(line) < 3:
            return True

        words = line.split()

        # Fragmented text (many single characters)
        if len(words) > 5:
            single_chars = sum(1 for w in words if len(w) <= 2)
            if single_chars / len(words) > 0.7:
                return True

        # Table of contents pattern
        if re.match(r'^.{5,50}\.{5,}\s*\d+$', line):
            return True

        # Page numbers
        if re.match(r'^(page|p\.?)\s*\d+|^\d+\s*(of|/)\s*\d+$', line, re.I):
            return True

        # Pure numbers/dates
        if re.match(r'^[\d\s\.\-\/]+$', line) and len(line) < 15:
            return True

        # Numeric tables
        if len(words) > 8:
            num_count = sum(1 for w in words if re.match(
                r'^\d+[\.\,]?\d*$', w))
            if num_count / len(words) > 0.6:
                return True

        # URLs
        if re.match(r'^https?://\S+$', line, re.I):
            return True

        # Spam characters
        spam_count = sum(1 for c in line if c in self._spam_chars_set)
        if spam_count > 5:
            return True

        # Dotted lines
        if len(line) > 10 and line.count('.') / len(line) > 0.4:
            return True

        # Name lists (all capitalized words)
        if re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+){10,}$', line):
            return True

        return False

    def _clean_artifacts(self, text: str) -> str:
        """Clean visual artifacts from text."""
        # Fix split uppercase words
        text = re.sub(
            r'\b([A-ZÄÖÜ])\s+([A-ZÄÖÜ])\s+([A-ZÄÖÜ])', r'\1\2\3', text)

        # Normalize repeated punctuation
        text = re.sub(r'\.{5,}', '...', text)
        text = re.sub(r'([.\-–—•·])\1{3,}', r'\1\1', text)

        # Normalize whitespace
        text = re.sub(r'\s{3,}', ' ', text)

        return text.strip()

    def _detect_severe_repetition(self, text: str) -> bool:
        """Detect if text has severe word repetition (likely extraction error)."""
        words = text.split()
        if len(words) < 10:
            return False

        # Check for 5+ consecutive identical words
        for i in range(len(words) - 4):
            if words[i] == words[i+1] == words[i+2] == words[i+3] == words[i+4]:
                return True

        # Check for one word dominating the text
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'for',
            'on', 'with', 'is', 'are', 'was', 'were', 'be', 'been'
        }
        word_counts = Counter(w.lower()
                              for w in words if w.lower() not in stopwords)

        if not word_counts:
            return False

        most_common_word, most_common_count = word_counts.most_common(1)[0]
        content_words = sum(word_counts.values())

        return content_words > 0 and (most_common_count / content_words) > 0.3

    def _remove_repetitions(self, text: str, max_repeat: int = 2) -> str:
        """Remove excessive repetitions from text."""
        # Clean repeated spam characters
        for char in self._spam_chars_set:
            pattern = rf'(\s*{re.escape(char)}\s*){{3,}}'
            text = re.sub(pattern, f' {char} ', text)

        text = re.sub(r'\s+', ' ', text).strip()

        words = text.split()
        if len(words) < 4:
            return text

        # Remove consecutive duplicate words
        result = []
        i = 0
        while i < len(words):
            word = words[i]
            count = 1
            while i + count < len(words) and words[i + count] == word:
                count += 1
            result.extend([word] * min(count, max_repeat))
            i += count

        text = ' '.join(result)

        # Remove repeated n-grams
        words = text.split()
        for n in [4, 3, 2]:
            if len(words) < n * 2:
                continue

            cleaned_words = []
            i = 0
            while i < len(words):
                if i + n * 2 <= len(words):
                    ngram = ' '.join(words[i:i+n])
                    next_ngram = ' '.join(words[i+n:i+n*2])
                    if ngram == next_ngram:
                        cleaned_words.extend(words[i:i+n])
                        j = i + n
                        while j + n <= len(words) and ' '.join(words[j:j+n]) == ngram:
                            j += n
                        i = j
                    else:
                        cleaned_words.append(words[i])
                        i += 1
                else:
                    cleaned_words.append(words[i])
                    i += 1
            words = cleaned_words

        return ' '.join(words)

    # -------------------------------------------------------------------------
    # Chunking
    # -------------------------------------------------------------------------

    def chunk_text(self, text: str, method: str = "semantic") -> List[str]:
        """
        Split text into chunks.

        Args:
            text: Cleaned text to chunk
            method: "semantic" (sentence-aware) or "simple" (character-based)

        Returns:
            List of text chunks
        """
        if method == "simple":
            return self._chunk_simple(text)
        else:
            return self._chunk_semantic(text)

    def _chunk_semantic(self, text: str) -> List[str]:
        """Sentence-aware chunking (from BERT pipeline)."""
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return []

        min_chars = self.config.min_chunk_chars
        max_chars = self.config.max_chunk_chars

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para_len = len(para)

            if min_chars <= para_len <= max_chars:
                # Paragraph fits as a chunk
                if current_chunk:
                    merged = current_chunk + " " + para
                    if len(merged) <= max_chars:
                        current_chunk = merged
                    else:
                        chunks.append(current_chunk)
                        current_chunk = para
                else:
                    current_chunk = para

            elif para_len > max_chars:
                # Paragraph too long - split by sentences
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                sentences = self._split_sentences(para)
                temp_chunk = ""

                for sent in sentences:
                    if not temp_chunk:
                        temp_chunk = sent
                    elif len(temp_chunk) + len(sent) + 1 <= max_chars:
                        temp_chunk += " " + sent
                    else:
                        if len(temp_chunk) >= min_chars:
                            chunks.append(temp_chunk)
                        temp_chunk = sent

                if temp_chunk:
                    if len(temp_chunk) >= min_chars:
                        chunks.append(temp_chunk)
                    else:
                        current_chunk = temp_chunk
            else:
                # Paragraph too short - accumulate
                if current_chunk:
                    current_chunk += " " + para
                else:
                    current_chunk = para

        if current_chunk and len(current_chunk) >= min_chars:
            chunks.append(current_chunk)

        return chunks

    def _chunk_simple(self, text: str) -> List[str]:
        """Simple character-based chunking with overlap."""
        chunk_size = self.config.langchain_chunk_size
        overlap = self.config.langchain_chunk_overlap

        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end in last 20% of chunk
                search_start = end - int(chunk_size * 0.2)
                search_text = text[search_start:end]

                for punct in ['. ', '! ', '? ', '\n\n', '\n']:
                    last_punct = search_text.rfind(punct)
                    if last_punct != -1:
                        end = search_start + last_punct + len(punct)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spacy."""
        nlp = _get_nlp()
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    # -------------------------------------------------------------------------
    # Metadata Extraction
    # -------------------------------------------------------------------------

    def extract_metadata(self, pdf_path: Union[str, Path]) -> Dict:
        """
        Extract metadata from PDF file path.

        Expected naming convention: COMPANYID_YEAR_reporttype.pdf
        Expected folder structure: reports/CompanyName/...

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with company_name, company_id, year, report_type
        """
        pdf_path = Path(pdf_path)
        filename = pdf_path.stem
        parts = pdf_path.parts

        # Extract company name from folder structure
        company_name = None
        for i, part in enumerate(parts):
            if part.lower() == 'reports' and i + 1 < len(parts):
                company_name = parts[i + 1]
                break

        if not company_name:
            skip_folders = {
                'factsheets', 'highlights', 'annual',
                'reports', 'data', 'pdfs'
            }
            for parent in [pdf_path.parent, pdf_path.parent.parent]:
                if parent.name.lower() not in skip_folders and parent.name:
                    company_name = parent.name
                    break

        # Parse filename
        name_parts = filename.split("_")

        company_id = None
        match = re.match(r'^(\d{2,3})_', filename)
        if match:
            company_id = match.group(1)
        elif name_parts:
            company_id = name_parts[0]

        year = None
        year_matches = re.findall(r'(20[12]\d)', filename)
        if year_matches:
            year = int(year_matches[0])
        elif len(name_parts) > 1:
            try:
                year = int(name_parts[1])
            except ValueError:
                pass

        report_type = None
        if len(name_parts) > 2:
            report_type = "_".join(name_parts[2:])

        return {
            "company_name": company_name,
            "company_id": company_id,
            "year": year,
            "report_type": report_type,
        }

    # -------------------------------------------------------------------------
    # Language Detection
    # -------------------------------------------------------------------------

    def detect_language(self, text: str) -> str:
        """
        Detect the language of the text.

        Uses multiple samples from different parts of the text
        for more reliable detection.

        Args:
            text: Text to analyze

        Returns:
            ISO language code (e.g., 'en', 'de', 'fr')
        """
        if not self.config.detect_language:
            return 'en'

        try:
            text_len = len(text)
            samples = []

            if text_len > 15000:
                # Sample from beginning, middle, and end
                samples = [
                    text[1000:4000],
                    text[text_len // 2:text_len // 2 + 3000],
                    text[-4000:-1000]
                ]
            else:
                samples = [text[100:min(5000, text_len - 100)]]

            votes = []
            for sample in samples:
                # Clean sample for better detection
                sample_clean = re.sub(r'[^\w\s]', ' ', sample)
                sample_clean = re.sub(r'\s+', ' ', sample_clean)

                if len(sample_clean) > 100:
                    lang, _ = langid.classify(sample_clean)
                    votes.append(lang)

            if votes:
                return Counter(votes).most_common(1)[0][0]

            return 'en'

        except Exception:
            return 'en'

    # -------------------------------------------------------------------------
    # Main Processing Methods
    # -------------------------------------------------------------------------

    def process_pdf(
        self,
        pdf_path: Union[str, Path],
        chunk_method: str = "semantic"
    ) -> ProcessedDocument:
        """
        Process a single PDF file.

        Args:
            pdf_path: Path to PDF file
            chunk_method: "semantic" or "simple"

        Returns:
            ProcessedDocument object
        """
        pdf_path = Path(pdf_path)

        # Extract metadata
        metadata = self.extract_metadata(pdf_path)

        # Extract text
        raw_text, num_pages = self.extract_text_from_pdf(pdf_path)

        if not raw_text:
            return ProcessedDocument(
                source_path=str(pdf_path),
                filename=pdf_path.name,
                **metadata,
                num_pages=num_pages,
            )

        # Clean text
        cleaned_text = self.clean_text(raw_text)

        # Detect language
        language = self.detect_language(cleaned_text)

        # Chunk text
        chunks = self.chunk_text(cleaned_text, method=chunk_method)

        return ProcessedDocument(
            source_path=str(pdf_path),
            filename=pdf_path.name,
            company_name=metadata["company_name"],
            company_id=metadata["company_id"],
            year=metadata["year"],
            report_type=metadata["report_type"],
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            chunks=chunks,
            language=language,
            num_pages=num_pages,
            extraction_method="pymupdf_blocks",
        )

    def process_folder(
        self,
        folder_path: Union[str, Path],
        chunk_method: str = "semantic",
        recursive: bool = True,
        show_progress: bool = True,
    ) -> List[ProcessedDocument]:
        """
        Process all PDFs in a folder.

        Args:
            folder_path: Path to folder containing PDFs
            chunk_method: "semantic" or "simple"
            recursive: Search subdirectories
            show_progress: Show progress bar

        Returns:
            List of ProcessedDocument objects
        """
        folder_path = Path(folder_path)

        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Find all PDFs
        if recursive:
            pdf_files = sorted(folder_path.rglob("*.pdf"))
        else:
            pdf_files = sorted(folder_path.glob("*.pdf"))

        if not pdf_files:
            print(f"⚠️ No PDFs found in {folder_path}")
            return []

        print(f"\n{'='*60}")
        print("PDF PREPROCESSING")
        print(f"{'='*60}")
        print(f"Folder: {folder_path}")
        print(f"PDFs found: {len(pdf_files)}")
        print(f"Chunk method: {chunk_method}")

        documents = []
        iterator = tqdm(
            pdf_files, desc="Processing") if show_progress else pdf_files

        for pdf_file in iterator:
            try:
                doc = self.process_pdf(pdf_file, chunk_method=chunk_method)
                documents.append(doc)
            except Exception as e:
                print(f"\n⚠️ Error processing {pdf_file.name}: {e}")
                continue

        total_chunks = sum(len(d.chunks) for d in documents)
        print(f"\n✓ Processed {len(documents)} documents")
        print(f"✓ Total chunks: {total_chunks}")

        return documents

    # -------------------------------------------------------------------------
    # Conversion Methods
    # -------------------------------------------------------------------------

    def get_langchain_chunks(
        self,
        documents: List[ProcessedDocument]
    ) -> List["DocumentChunk"]:
        """
        Convert processed documents to DocumentChunk objects.

        Args:
            documents: List of ProcessedDocument objects

        Returns:
            List of DocumentChunk objects
        """
        all_chunks = []

        for doc in documents:
            for i, chunk_text in enumerate(doc.chunks):
                chunk_id = f"{doc.company_id}_{i:03d}" if doc.company_id else f"chunk_{i:03d}"

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source_path=doc.source_path,
                    filename=doc.filename,
                    company_name=doc.company_name,
                    company_id=doc.company_id,
                    year=str(doc.year) if doc.year else None,
                    report_type=doc.report_type,
                    language=doc.language,
                    chunk_index=i,
                )
                all_chunks.append(chunk)

        return all_chunks

    def to_langchain_documents(
        self,
        documents: List[ProcessedDocument]
    ) -> List:
        """
        Convert to LangChain Document format for RAG pipeline.

        Args:
            documents: List of ProcessedDocument objects

        Returns:
            List of LangChain Document objects
        """
        chunks = self.get_langchain_chunks(documents)
        return [chunk.to_langchain_document() for chunk in chunks]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def preprocess_pdfs(
    folder_path: str,
    chunk_method: str = "semantic",
    config: Optional[PreprocessingConfig] = None,
) -> List[ProcessedDocument]:
    """
    Convenience function to preprocess PDFs in a folder.

    Args:
        folder_path: Path to folder containing PDFs
        chunk_method: "semantic" or "simple"
        config: Optional custom configuration

    Returns:
        List of ProcessedDocument objects
    """
    preprocessor = PDFPreprocessor(config)
    return preprocessor.process_folder(folder_path, chunk_method=chunk_method)


def preprocess_single_pdf(
    pdf_path: str,
    chunk_method: str = "semantic",
    config: Optional[PreprocessingConfig] = None,
) -> ProcessedDocument:
    """
    Convenience function to preprocess a single PDF.

    Args:
        pdf_path: Path to PDF file
        chunk_method: "semantic" or "simple"
        config: Optional custom configuration

    Returns:
        ProcessedDocument object
    """
    preprocessor = PDFPreprocessor(config)
    return preprocessor.process_pdf(pdf_path, chunk_method=chunk_method)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("PDF Preprocessing Module")
    print("=" * 60)
    print("\nUsage:")
    print("  from preprocessing import PDFPreprocessor, preprocess_pdfs")
    print()
    print("  # Quick start")
    print("  docs = preprocess_pdfs('../data/reports')")
    print()
    print("  # With custom config")
    print("  from preprocessing import PreprocessingConfig")
    print("  config = PreprocessingConfig(min_chunk_chars=400, max_chunk_chars=1200)")
    print("  preprocessor = PDFPreprocessor(config)")
    print("  docs = preprocessor.process_folder('../data/reports')")
    print()
    print("  # Get LangChain-compatible documents")
    print("  lc_docs = preprocessor.to_langchain_documents(docs)")
