"""
PDF Preprocessing Module
========================

Shared preprocessing utilities for PDF extraction, cleaning, chunking, and translation.
Used by both RAG and BERT analysis pipelines.

Usage:
    from preprocessing import PDFPreprocessor, PreprocessingConfig

    preprocessor = PDFPreprocessor()
    documents = preprocessor.process_folder("../data/reports")
"""

import os
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import fitz  # PyMuPDF
import langid
import spacy
import torch
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

from .gpu_utils import GPUManager
import logging
logging.getLogger("fitz").setLevel(logging.ERROR)

# Lazy-loaded spacy
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
# TRANSLATION MODELS
# =============================================================================

HELSINKI_MODELS = {
    "de": "Helsinki-NLP/opus-mt-de-en",
    "es": "Helsinki-NLP/opus-mt-es-en",
    "it": "Helsinki-NLP/opus-mt-it-en",
    "zh": "Helsinki-NLP/opus-mt-zh-en",
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "nl": "Helsinki-NLP/opus-mt-nl-en",
    "pt": "Helsinki-NLP/opus-mt-ROMANCE-en",
    "pl": "Helsinki-NLP/opus-mt-pl-en",
    "ru": "Helsinki-NLP/opus-mt-ru-en",
    "ja": "Helsinki-NLP/opus-mt-ja-en",
    "ko": "Helsinki-NLP/opus-mt-ko-en",
}


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

    # Translation settings
    translate_batch_size: int = 16
    translate_max_input_tokens: int = 512
    translate_max_output_tokens: int = 512


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ProcessedDocument:
    """Represents a processed PDF document."""

    source_path: str
    filename: str

    # Metadata
    company_name: Optional[str] = None
    company_id: Optional[str] = None
    year: Optional[int] = None
    report_type: Optional[str] = None

    # Content
    raw_text: str = ""
    cleaned_text: str = ""
    chunks: List[str] = field(default_factory=list)
    original_chunks: List[str] = field(default_factory=list)

    # Language
    language: str = "en"
    translated: bool = False

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
            "translated": self.translated,
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
    translated: bool = False
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
                "translated": self.translated,
                "chunk_index": self.chunk_index,
            },
        )


# =============================================================================
# TRANSLATOR CLASS
# =============================================================================

class Translator:
    """Handles translation using Helsinki-NLP models with GPU support."""

    def __init__(self, config: PreprocessingConfig, quiet: bool = False):
        self.config = config
        self.quiet = quiet
        self.gpu = GPUManager()
        self._model = None
        self._tokenizer = None
        self._current_lang = None

        if not quiet:
            print(f"✓ Translator {self.gpu}")

    def _load_model(self, src_lang: str):
        """Load translation model for source language."""
        if self._current_lang == src_lang and self._model is not None:
            return

        self._unload()

        model_name = HELSINKI_MODELS.get(src_lang)
        if not model_name:
            raise ValueError(f"No translation model for language: {src_lang}")

        self._tokenizer = MarianTokenizer.from_pretrained(model_name)
        self._model = MarianMTModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.gpu.is_cuda else torch.float32,
        )
        self._model.to(self.gpu.device)
        self._model.eval()
        self._current_lang = src_lang

    def _unload(self):
        """Unload current model to free memory."""
        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._current_lang = None
        self.gpu.clear()

    def translate(self, texts: List[str], src_lang: str) -> List[str]:
        """Translate texts from source language to English."""
        if src_lang == "en":
            return texts

        if src_lang not in HELSINKI_MODELS:
            if not self.quiet:
                print(
                    f"⚠️ No translation model for '{src_lang}', keeping original")
            return texts

        self._load_model(src_lang)

        translated = []
        batch_size = self.config.translate_batch_size

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i: i + batch_size]
                try:
                    inputs = self._tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.config.translate_max_input_tokens,
                    )
                    inputs = {k: v.to(self.gpu.device)
                              for k, v in inputs.items()}

                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=self.config.translate_max_output_tokens,
                        no_repeat_ngram_size=3,
                        repetition_penalty=1.2,
                        num_beams=4,
                        early_stopping=True,
                        do_sample=False,
                        use_cache=True,
                    )

                    batch_translations = self._tokenizer.batch_decode(
                        outputs, skip_special_tokens=True
                    )
                    translated.extend(batch_translations)
                    del inputs, outputs

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        self.gpu.clear()
                        # Fallback to single-item processing
                        for chunk in batch:
                            try:
                                inputs = self._tokenizer(
                                    [chunk],
                                    return_tensors="pt",
                                    truncation=True,
                                    max_length=self.config.translate_max_input_tokens,
                                )
                                inputs = {k: v.to(self.gpu.device)
                                          for k, v in inputs.items()}
                                output = self._model.generate(
                                    **inputs,
                                    max_new_tokens=self.config.translate_max_output_tokens,
                                    no_repeat_ngram_size=3,
                                    num_beams=1,
                                )
                                translated.append(
                                    self._tokenizer.decode(
                                        output[0], skip_special_tokens=True)
                                )
                                del inputs, output
                            except Exception:
                                translated.append(chunk)
                                self.gpu.clear()
                    else:
                        raise

        return translated

    def cleanup(self):
        """Release all resources."""
        self._unload()


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
    - Translation to English (optional)
    - Metadata extraction from file paths
    - Language detection
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None, quiet: bool = False):
        self.config = config or PreprocessingConfig()
        self.quiet = quiet
        self._spam_chars_set = set(self.config.spam_chars)
        self._translator: Optional[Translator] = None

    @property
    def translator(self) -> Translator:
        """Lazy-load translator."""
        if self._translator is None:
            self._translator = Translator(self.config, quiet=self.quiet)
        return self._translator

    # -------------------------------------------------------------------------
    # PDF Text Extraction
    # -------------------------------------------------------------------------

    def extract_text_from_pdf(self, pdf_path: Union[str, Path]) -> Tuple[str, int]:
        """Extract text from PDF using PyMuPDF with table detection."""
        pdf_path = Path(pdf_path)
        all_text = []
        num_pages = 0

        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
        except Exception as e:
            if not self.quiet:
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
                except Exception:
                    pass

            # Try structured extraction first
            try:
                blocks = page.get_text("dict", sort=True)["blocks"]
            except Exception:
                text = page.get_text("text")
                if text:
                    all_text.append(text)
                continue

            page_paragraphs = []
            for block in blocks:
                if block.get("type") != 0:
                    continue

                bbox = block.get("bbox", [0, 0, 0, 0])
                block_height = bbox[3] - bbox[1]

                if block_height < 5:
                    continue

                if self.config.skip_tables:
                    is_in_table = any(self._bbox_overlap(bbox, tb)
                                      for tb in table_bboxes)
                    if is_in_table:
                        continue

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
                    paragraph = re.sub(r"\s+", " ", paragraph).strip()
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
        """Clean and preprocess extracted text."""
        text = self._fix_encoding(raw_text)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        lines = []
        for line in text.split("\n"):
            line = re.sub(r"[ \t]+", " ", line).strip()
            if not line:
                lines.append("")
            elif self._is_noise_line(line):
                continue
            else:
                lines.append(line)

        paragraphs = []
        current = []

        for i, line in enumerate(lines):
            if line:
                current.append(line)
                ends_sentence = re.search(r"[.!?]\s*$", line)
                if ends_sentence and (i + 1 >= len(lines) or not lines[i + 1]):
                    para = " ".join(current)
                    para = self._clean_artifacts(para)
                    if len(para) > self.config.min_paragraph_chars:
                        if not self._detect_severe_repetition(para):
                            paragraphs.append(para)
                    current = []
            elif current:
                if current and re.search(r"[.!?]\s*$", current[-1]):
                    para = " ".join(current)
                    para = self._clean_artifacts(para)
                    if len(para) > self.config.min_paragraph_chars:
                        if not self._detect_severe_repetition(para):
                            paragraphs.append(para)
                    current = []

        if current:
            para = " ".join(current)
            para = self._clean_artifacts(para)
            if len(para) > self.config.min_paragraph_chars:
                if not self._detect_severe_repetition(para):
                    paragraphs.append(para)

        return "\n\n".join(paragraphs)

    def _fix_encoding(self, text: str) -> str:
        """Fix common PDF encoding issues."""
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"\(cid:\d+\)", "", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

        for char in ["\u00ad", "\u200b", "\u200c", "\u200d", "\ufeff"]:
            text = text.replace(char, "")

        return text

    def _is_noise_line(self, line: str) -> bool:
        """Check if a line is noise (headers, footers, page numbers, etc.)."""
        line = line.strip()

        if len(line) < 3:
            return True

        words = line.split()

        if len(words) > 5:
            single_chars = sum(1 for w in words if len(w) <= 2)
            if single_chars / len(words) > 0.7:
                return True

        if re.match(r"^.{5,50}\.{5,}\s*\d+$", line):
            return True

        if re.match(r"^(page|p\.?)\s*\d+|^\d+\s*(of|/)\s*\d+$", line, re.I):
            return True

        if re.match(r"^[\d\s\.\-\/]+$", line) and len(line) < 15:
            return True

        if len(words) > 8:
            num_count = sum(1 for w in words if re.match(
                r"^\d+[\.\,]?\d*$", w))
            if num_count / len(words) > 0.6:
                return True

        if re.match(r"^https?://\S+$", line, re.I):
            return True

        spam_count = sum(1 for c in line if c in self._spam_chars_set)
        if spam_count > 5:
            return True

        if len(line) > 10 and line.count(".") / len(line) > 0.4:
            return True

        if re.match(r"^[A-Z][a-z]+(\s+[A-Z][a-z]+){10,}$", line):
            return True

        return False

    def _clean_artifacts(self, text: str) -> str:
        """Clean visual artifacts from text."""
        text = re.sub(
            r"\b([A-ZÄÖÜ])\s+([A-ZÄÖÜ])\s+([A-ZÄÖÜ])", r"\1\2\3", text)
        text = re.sub(r"\.{5,}", "...", text)
        text = re.sub(r"([.\-–—•·])\1{3,}", r"\1\1", text)
        text = re.sub(r"\s{3,}", " ", text)
        return text.strip()

    def _detect_severe_repetition(self, text: str) -> bool:
        """Detect if text has severe word repetition (likely extraction error)."""
        words = text.split()
        if len(words) < 10:
            return False

        for i in range(len(words) - 4):
            if words[i] == words[i + 1] == words[i + 2] == words[i + 3] == words[i + 4]:
                return True

        stopwords = {
            "the", "a", "an", "and", "or", "of", "to", "in", "for",
            "on", "with", "is", "are", "was", "were", "be", "been",
        }
        word_counts = Counter(w.lower()
                              for w in words if w.lower() not in stopwords)

        if not word_counts:
            return False

        most_common_word, most_common_count = word_counts.most_common(1)[0]
        content_words = sum(word_counts.values())

        return content_words > 0 and (most_common_count / content_words) > 0.3

    # -------------------------------------------------------------------------
    # Chunking
    # -------------------------------------------------------------------------

    def chunk_text(self, text: str, method: str = "semantic") -> List[str]:
        """Split text into chunks."""
        if method == "simple":
            return self._chunk_simple(text)
        return self._chunk_semantic(text)

    def _chunk_semantic(self, text: str) -> List[str]:
        """Sentence-aware chunking."""
        paragraphs = re.split(r"\n\s*\n", text)
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

            if end < len(text):
                search_start = end - int(chunk_size * 0.2)
                search_text = text[search_start:end]

                for punct in [". ", "! ", "? ", "\n\n", "\n"]:
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
        """Extract metadata from PDF file path."""
        pdf_path = Path(pdf_path)
        filename = pdf_path.stem
        parts = pdf_path.parts

        company_name = None
        for i, part in enumerate(parts):
            if part.lower() == "reports" and i + 1 < len(parts):
                company_name = parts[i + 1]
                break

        if not company_name:
            for parent in [pdf_path.parent, pdf_path.parent.parent]:
                if parent.name.lower() and parent.name:
                    company_name = parent.name
                    break

        name_parts = filename.split("_")

        company_id = None
        match = re.match(r"^(\d{2,3})_", filename)
        if match:
            company_id = match.group(1)
        elif name_parts:
            company_id = name_parts[0]

        year = None
        year_matches = re.findall(r"(20[12]\d)", filename)
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

    def detect_language(self, chunks: List[str]) -> str:
        """Detect the language from the provided text chunks."""
        if not self.config.detect_language:
            return "en"

        try:
            votes = []
            for chunk in chunks:
                if len(chunk) > 100:
                    lang, _ = langid.classify(chunk)
                    votes.append(lang)

            if votes:
                return Counter(votes).most_common(1)[0][0]
            return "en"

        except Exception:
            return "en"

    # -------------------------------------------------------------------------
    # Main Processing Methods
    # -------------------------------------------------------------------------

    def process_pdf(
        self,
        pdf_path: Union[str, Path],
        chunk_method: str = "semantic",
        translate_to_english: bool = True,
        show_progress: bool = True,
        step_callback: Optional[Callable[[str, int], None]] = None,
    ) -> ProcessedDocument:
        """
        Process a single PDF file.

        Args:
            pdf_path: Path to PDF file
            chunk_method: "semantic" or "simple"
            translate_to_english: Whether to translate non-English text
            show_progress: Show tqdm progress bar (disabled when using step_callback)
            step_callback: Optional callback function(step_name, step_num) for external progress
        """
        pdf_path = Path(pdf_path)

        # Use either tqdm or callback for progress
        use_tqdm = show_progress and step_callback is None and not self.quiet

        def update_progress(step_num: int, step_name: str):
            if step_callback:
                step_callback(step_name, step_num)
            elif use_tqdm and pbar:
                pbar.set_postfix_str(step_name)
                pbar.update(1)

        pbar = None
        if use_tqdm:
            pbar = tqdm(
                total=5, desc=f"Processing {pdf_path.name}", unit="step")

        try:
            # Step 1: Extract text
            update_progress(1, "Extract")
            metadata = self.extract_metadata(pdf_path)
            raw_text, num_pages = self.extract_text_from_pdf(pdf_path)

            if not raw_text:
                if pbar:
                    pbar.close()
                return ProcessedDocument(
                    source_path=str(pdf_path),
                    filename=pdf_path.name,
                    **metadata,
                    num_pages=num_pages,
                )

            # Step 2: Clean text
            update_progress(2, "Clean")
            cleaned_text = self.clean_text(raw_text)

            # Step 3: Chunk text
            update_progress(3, "Chunk")
            chunks = self.chunk_text(cleaned_text, method=chunk_method)

            # Step 4: Detect language
            update_progress(4, "Lang")
            sample_size = min(5, len(chunks))
            language = self.detect_language(random.sample(
                chunks, sample_size)) if chunks else "en"

            # Step 5: Translation
            original_chunks = chunks.copy()
            translated = False

            if translate_to_english and language != "en" and language in HELSINKI_MODELS:
                update_progress(5, f"Translate")
                chunks = self.translator.translate(chunks, language)
                translated = True
            else:
                update_progress(5, "NoTranslate")

        finally:
            if pbar:
                pbar.close()

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
            original_chunks=original_chunks if translated else [],
            language=language,
            translated=translated,
            num_pages=num_pages,
            extraction_method="pymupdf_blocks",
        )

    def process_folder(
        self,
        folder_path: Union[str, Path],
        chunk_method: str = "semantic",
        translate_to_english: bool = True,
        recursive: bool = True,
        show_progress: bool = True,
    ) -> List[ProcessedDocument]:
        """Process all PDFs in a folder."""
        folder_path = Path(folder_path)

        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        if recursive:
            pdf_files = sorted(folder_path.rglob("*.pdf"))
        else:
            pdf_files = sorted(folder_path.glob("*.pdf"))

        if not pdf_files:
            if not self.quiet:
                print(f"⚠️ No PDFs found in {folder_path}")
            return []

        if not self.quiet:
            print(f"\n{'='*60}")
            print("PDF PREPROCESSING")
            print(f"{'='*60}")
            print(f"Folder: {folder_path}")
            print(f"PDFs found: {len(pdf_files)}")
            print(f"Chunk method: {chunk_method}")
            print(
                f"Translation: {'enabled' if translate_to_english else 'disabled'}")

        documents = []
        iterator = tqdm(
            pdf_files, desc="Processing") if show_progress and not self.quiet else pdf_files

        for pdf_file in iterator:
            try:
                doc = self.process_pdf(
                    pdf_file,
                    chunk_method=chunk_method,
                    translate_to_english=translate_to_english,
                    show_progress=False,  # Disable per-file progress when doing folder
                )
                documents.append(doc)
            except Exception as e:
                if not self.quiet:
                    print(f"\n⚠️ Error processing {pdf_file.name}: {e}")
                continue

        total_chunks = sum(len(d.chunks) for d in documents)
        translated_count = sum(1 for d in documents if d.translated)

        if not self.quiet:
            print(f"\n✓ Processed {len(documents)} documents")
            print(f"✓ Total chunks: {total_chunks}")
            if translate_to_english:
                print(f"✓ Translated: {translated_count} documents")

        return documents

    def cleanup(self):
        """Release all resources."""
        if self._translator is not None:
            self._translator.cleanup()
            self._translator = None

    # -------------------------------------------------------------------------
    # Conversion Methods
    # -------------------------------------------------------------------------

    def get_langchain_chunks(self, documents: List[ProcessedDocument]) -> List[DocumentChunk]:
        """Convert processed documents to DocumentChunk objects."""
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
                    translated=doc.translated,
                    chunk_index=i,
                )
                all_chunks.append(chunk)

        return all_chunks

    def to_langchain_documents(self, documents: List[ProcessedDocument]) -> List:
        """Convert to LangChain Document format for RAG pipeline."""
        chunks = self.get_langchain_chunks(documents)
        return [chunk.to_langchain_document() for chunk in chunks]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def preprocess_pdfs(
    folder_path: str,
    chunk_method: str = "semantic",
    translate_to_english: bool = True,
    config: Optional[PreprocessingConfig] = None,
) -> List[ProcessedDocument]:
    """Convenience function to preprocess PDFs in a folder."""
    preprocessor = PDFPreprocessor(config)
    try:
        return preprocessor.process_folder(
            folder_path,
            chunk_method=chunk_method,
            translate_to_english=translate_to_english,
        )
    finally:
        preprocessor.cleanup()


def preprocess_single_pdf(
    pdf_path: str,
    chunk_method: str = "semantic",
    translate_to_english: bool = True,
    config: Optional[PreprocessingConfig] = None,
) -> ProcessedDocument:
    """Convenience function to preprocess a single PDF."""
    preprocessor = PDFPreprocessor(config)
    try:
        return preprocessor.process_pdf(
            pdf_path,
            chunk_method=chunk_method,
            translate_to_english=translate_to_english,
        )
    finally:
        preprocessor.cleanup()


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
    print("  # Without translation")
    print("  docs = preprocess_pdfs('../data/reports', translate_to_english=False)")
