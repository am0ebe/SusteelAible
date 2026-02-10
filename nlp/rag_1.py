"""
RAG Pipeline for Company Decarbonisation Report Analysis
========================================================

This module provides a pipeline for:
1. Loading preprocessed data from JSON cache files (bert.json with ClimateBERT filtering)
2. Extracting barriers and motivators using LLM extraction
3. Aggregating results by company and year

Strategy: ClimateBERT filters → Group by company-year → Extract per group → Save

Usage:
    from rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()
    pipeline.load_from_cache()
    pipeline.extract_all_companies()
"""

import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Literal

from nlp.gpu_utils import GPUManager
from nlp.data_loader import load_prep_cache, load_bert_cache

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from tabulate import tabulate

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
load_dotenv()


os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RAGConfig:
    """Configuration for RAG pipeline. See rag_test.py for model options."""

    # LLM (REQUIRED)
    llm_provider: Literal["ollama", "groq"]
    model: str

    # LLM optional
    llm_temperature: float = 0.0
    ollama_base_url: str = "http://localhost:11434"

    # Context window size (tokens). Ollama: actual context. Cloud APIs: batch size calc only.
    # Capped at 32k for batch_size calculation — larger windows don't improve extraction quality.
    ctx: int = 8096

    # Stop sequences to halt generation early
    # Extensions: "Note:", "Explanation:", "Here are", "I found"
    stop_tokens: tuple = ("NONE_FOUND\n", "\n\n\n")

    # Pipeline
    use_bert_cache: bool = True
    cache_dir: str = "../cache"
    output_folder: str = "../out"
    # Auto-calculated from ctx (chunks per LLM call)
    batch_size: Optional[int] = None
    min_detector_score: float = 0.0

    # Max ctx used for batch_size calc. Cloud APIs have huge windows (1M+),
    # but batches >~70 chunks hurt extraction quality (model loses focus).
    _BATCH_CTX_CAP: int = 32000

    def __post_init__(self):
        """Auto-calculate batch_size from context window if not set."""
        if self.batch_size is None:
            # Cap ctx for batch calc — huge windows don't improve extraction
            effective_ctx = min(self.ctx, self._BATCH_CTX_CAP)

            # Calculate prompt overhead from actual template
            prompt_chars = len(BARRIER_MAP_PROMPT.messages[0].prompt.template)
            prompt_tokens = prompt_chars // 4  # ~4 chars per token

            # Estimates (can't calc without loading data first):
            # - avg_chunk_tokens: chunks are 600-1600 chars, avg ~1100 chars = ~275 tokens
            #   Using 400 as conservative estimate for longer chunks
            # - output_per_chunk: each extraction is ~120 chars = ~30 tokens
            #   Not all chunks produce output (~30%), but reserve space for high-yield batches
            avg_chunk_tokens = 400
            output_per_chunk = 30
            safety_margin = 500

            available = effective_ctx - prompt_tokens - safety_margin
            tokens_per_chunk = avg_chunk_tokens + output_per_chunk
            self.batch_size = max(1, int(available / tokens_per_chunk))


# =============================================================================
# PROMPTS
# =============================================================================

# OLD PROMPTS (caused hallucination - model reused example IDs like 012_003)
# BARRIER_MAP_PROMPT_OLD = ChatPromptTemplate.from_template("""Extract BARRIERS to decarbonisation from {company} ({year}) report.
#
# BARRIER = challenge, constraint, risk, or factor that makes reducing GHG emissions harder.
#
# RULES:
# - Copy text EXACTLY as written (verbatim)
# - Each chunk starts with [chunk_id] - use that exact ID
#
# OUTPUT FORMAT (STRICT):
# [chunk_id]|||verbatim text
#
# EXAMPLE:
# [012_003]|||The high cost of green hydrogen limits our decarbonisation options.
# [012_007]|||Carbon capture technology remains expensive and unproven at scale.
#
# One per line. No other text. No JSON. No explanations. No quotes.
# If none found: NONE_FOUND
#
# TEXT:
# {context}""")

BARRIER_MAP_PROMPT = ChatPromptTemplate.from_template("""Extract BARRIERS to decarbonisation from {company} ({year}) report.

BARRIER = challenge, constraint, risk, or factor that makes reducing GHG emissions harder.

RULES:
- Copy text EXACTLY as written (verbatim)
- Each chunk in TEXT starts with [{company_id}_XXX] - use that EXACT ID
- Valid IDs start with {company_id}_ (e.g., {company_id}_001, {company_id}_042)
- NEVER invent IDs or use IDs from other companies

OUTPUT FORMAT:
[{company_id}_XXX]|||verbatim text

One per line. No explanations. If none found: NONE_FOUND

TEXT:
{context}""")

MOTIVATOR_MAP_PROMPT = ChatPromptTemplate.from_template("""Extract MOTIVATORS for decarbonisation from {company} ({year}) report.

MOTIVATOR = driver, incentive, commitment, or pressure that encourages reducing GHG emissions.

RULES:
- Copy text EXACTLY as written (verbatim)
- Each chunk in TEXT starts with [{company_id}_XXX] - use that EXACT ID
- Valid IDs start with {company_id}_ (e.g., {company_id}_001, {company_id}_042)
- NEVER invent IDs or use IDs from other companies

OUTPUT FORMAT:
[{company_id}_XXX]|||verbatim text

One per line. No explanations. If none found: NONE_FOUND

TEXT:
{context}""")


# =============================================================================
# MAIN PIPELINE CLASS
# =============================================================================

class RAGPipeline:
    """RAG pipeline for extracting barriers and motivators from sustainability reports."""

    def __init__(self, config: RAGConfig):
        """Initialize the pipeline."""
        self.config = config
        self.gpu = GPUManager()

        # Lazy-loaded LLM
        self._llm = None

        # Data
        self.chunks: List = []
        # (company_id, year) -> chunks
        self.grouped_chunks: Dict[Tuple[str, str], List] = {}

        print(f"✓ RAG Pipeline initialized ({self.gpu})")

    # -------------------------------------------------------------------------
    # Lazy-loaded properties
    # -------------------------------------------------------------------------

    @property
    def llm(self):
        """Lazy-load the LLM based on config.llm_provider."""
        if self._llm is None:
            if self.config.llm_provider == "groq":
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError(
                        "GROQ_API_KEY not found. Check .env file.")
                print(f"Loading Groq: {self.config.model}")
                groq_kwargs = {
                    "model": self.config.model,
                    "api_key": api_key,
                    "temperature": self.config.llm_temperature,
                    "stop_sequences": list(self.config.stop_tokens),
                }

                self._llm = ChatGroq(**groq_kwargs)

            else:  # ollama
                print(f"Loading Ollama: {self.config.model}")
                self._llm = ChatOllama(
                    model=self.config.model,
                    base_url=self.config.ollama_base_url,
                    temperature=self.config.llm_temperature,
                    num_ctx=self.config.ctx,
                    stop=list(self.config.stop_tokens),
                )
        return self._llm

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------

    def load_from_cache(self) -> List:
        """Load documents from JSON cache files and group by company-year."""
        source = 'bert.json' if self.config.use_bert_cache else 'prep.json'

        if self.config.use_bert_cache:
            self.chunks = load_bert_cache(
                cache_dir=self.config.cache_dir,
                as_langchain=True
            )
        else:
            self.chunks = load_prep_cache(
                cache_dir=self.config.cache_dir,
                as_langchain=True
            )

        if not self.chunks:
            raise ValueError(
                f"No chunks loaded from {source}. "
                f"Check cache files exist in {self.config.cache_dir}"
            )

        # Filter by detector score if threshold is set
        if self.config.min_detector_score > 0:
            original_count = len(self.chunks)
            self.chunks = [
                c for c in self.chunks
                if c.metadata.get("detector_score", 0) >= self.config.min_detector_score
            ]
            filtered_count = original_count - len(self.chunks)
            print(f"✓ Filtered {filtered_count} chunks below detector_score={self.config.min_detector_score} "
                  f"({len(self.chunks)} remaining)")

        # Group chunks by (company_id, year)
        self._group_chunks()

        companies = set(c.metadata.get("company_id")
                        for c in self.chunks if c.metadata.get("company_id"))
        print(
            f"✓ Loaded {len(self.chunks)} chunks from {len(companies)} companies ({self.config.cache_dir})")

        return self.chunks

    def _group_chunks(self):
        """Group chunks by (company_id, year) for batch processing."""
        self.grouped_chunks = defaultdict(list)

        for chunk in self.chunks:
            company_id = chunk.metadata.get("company_id")
            year = chunk.metadata.get("year")

            if company_id and year:
                self.grouped_chunks[(company_id, year)].append(chunk)

    # -------------------------------------------------------------------------
    # Overview / Statistics
    # -------------------------------------------------------------------------

    def _build_chunk_matrix(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Build matrix of chunk counts: rows=years, cols=companies."""
        # Collect data
        company_years = defaultdict(lambda: defaultdict(int))
        company_names = {}  # company_id -> company_name

        for chunk in self.chunks:
            company_id = chunk.metadata.get("company_id")
            company_name = chunk.metadata.get("company_name")
            year = chunk.metadata.get("year")

            if company_id and year:
                company_years[company_id][year] += 1
                if company_name:
                    company_names[company_id] = company_name

        if not company_years:
            return pd.DataFrame(), {}

        # Get all years and companies
        all_years = sorted(set(
            year for cy in company_years.values() for year in cy.keys()
        ))
        all_companies = sorted(company_years.keys())

        # Build matrix
        data = []
        for year in all_years:
            row = {"Year": year}
            for company_id in all_companies:
                row[company_id] = company_years[company_id].get(year, 0)
            data.append(row)

        df = pd.DataFrame(data)

        # Add totals row
        totals = {"Year": "TOTAL"}
        for company_id in all_companies:
            totals[company_id] = df[company_id].sum()
        df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

        # Add totals column
        df["TOTAL"] = df[all_companies].sum(axis=1)

        return df, company_names

    def print_chunk_overview(self):
        """Print chunk count matrix: years (rows) x companies (cols)."""
        df, company_names = self._build_chunk_matrix()

        if df.empty:
            print("No data to display.")
            return

        print(f"\n{'='*70}")
        print("CHUNK DISTRIBUTION (Years × Companies)")
        print(f"{'='*70}")

        # Create display names for columns
        display_df = df.copy()
        for col in display_df.columns:
            if col in company_names:
                # Shorten company name if needed
                name = company_names[col]
                if len(name) > 12:
                    name = name[:10] + ".."
                display_df = display_df.rename(
                    columns={col: f"{name}\n({col})"})

        print(tabulate(display_df, headers='keys',
              tablefmt='simple', showindex=False))
        print(f"{'='*70}\n")

    # -------------------------------------------------------------------------
    # Extraction
    # -------------------------------------------------------------------------

    def get_companies(self) -> List[str]:
        """Get sorted list of unique company IDs."""
        return sorted(set(key[0] for key in self.grouped_chunks.keys()))

    def get_years_for_company(self, company_id: str) -> List[str]:
        """Get sorted list of years for a specific company."""
        return sorted(set(
            key[1] for key in self.grouped_chunks.keys()
            if key[0] == company_id
        ))

    def _get_company_name(self, company_id: str) -> str:
        """Get company name from chunks metadata."""
        for chunk in self.chunks:
            if chunk.metadata.get("company_id") == company_id:
                name = chunk.metadata.get("company_name")
                if name:
                    return name
        return company_id

    # -------------------------------------------------------------------------
    # BERT Metadata Helpers
    # -------------------------------------------------------------------------

    def _get_chunk_metadata(self, chunk) -> Dict[str, Any]:
        """Extract BERT metadata dict from a chunk for output row."""
        return {
            "source_chunk_id": chunk.metadata.get("chunk_id"),
            "detector_score": chunk.metadata.get("detector_score", 0),
            "specificity_score": chunk.metadata.get("specificity_score", 0),
            "specificity_label": chunk.metadata.get("specificity_label"),
            "commitment_score": chunk.metadata.get("commitment_score", 0),
            "commitment_label": chunk.metadata.get("commitment_label"),
            "sentiment_label": chunk.metadata.get("sentiment_label"),
            "sentiment_score": chunk.metadata.get("sentiment_score", 0),
            "netzero_label": chunk.metadata.get("netzero_label"),
            "netzero_score": chunk.metadata.get("netzero_score", 0),
        }

    def _format_chunk(self, chunk) -> str:
        """Format chunk with chunk_id for prompt context.

        BERT labels are NOT included - they're looked up after extraction
        via chunk_id to avoid confusing the LLM.
        """
        chunk_id = chunk.metadata.get("chunk_id", "unknown")
        return f"[{chunk_id}]\n{chunk.page_content}"

    def _sort_chunks_by_quality(self, chunks: List) -> List:
        """Sort chunks by composite quality score (best first).
        """
        def score(c):
            return (
                c.metadata.get("detector_score", 0) * 0.4 +
                c.metadata.get("specificity_score", 0) * 0.4 +
                c.metadata.get("commitment_score", 0) * 0.2
            )
        return sorted(chunks, key=score, reverse=True)

    def _parse_llm_response(self, response_text: str) -> List[Tuple[str, str]]:
        """Parse [chunk_id]|||text format. No filtering - highest recall.

        Invalid chunk IDs are handled downstream (lookup warns + skips).
        Deduplication happens in rag2/topic modeling.
        """
        if not response_text or "NONE_FOUND" in response_text.upper():
            return []

        # Pattern: [chunk_id]|||text
        pattern = re.compile(r'^\[([^\]]+)\]\|\|\|(.+)$', re.MULTILINE)
        return [(m.group(1).strip(), m.group(2).strip()) for m in pattern.finditer(response_text)]

    def _map_extract(
        self,
        chunks: List,
        company: str,
        company_id: str,
        year: str,
        extract_type: str
    ) -> List[Tuple[str, str]]:
        """MAP phase: Extract barriers/motivators from a single company-year group.

        Returns list of (chunk_id, extracted_text) tuples.
        No filtering - invalid chunk IDs handled downstream via lookup.
        """
        if not chunks:
            return []

        # Sort by BERT quality score (best chunks first)
        chunks = self._sort_chunks_by_quality(chunks)

        # Select prompt
        prompt = BARRIER_MAP_PROMPT if extract_type == "barriers" else MOTIVATOR_MAP_PROMPT
        chain = prompt | self.llm

        batch_size = self.config.batch_size
        n_batches = (len(chunks) + batch_size - 1) // batch_size
        all_results = []

        # Process all chunks in batches
        label = extract_type[0].upper()  # B or M
        extract_start = time.time()

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(chunks))
            batch = chunks[start:end]

            # Build context with chunk IDs (no BERT labels - looked up after)
            context = "\n\n---\n\n".join(
                self._format_chunk(chunk)
                for chunk in batch
            )

            # Progress: overwrite line in-place
            elapsed = time.time() - extract_start
            print(f"    {label} {batch_idx+1}/{n_batches} | {len(all_results)} found | {elapsed:.0f}s", end="\r", flush=True)

            for attempt in range(3):
                try:
                    response = chain.invoke({
                        "company": company,
                        "company_id": company_id,
                        "year": year,
                        "context": context
                    })
                    raw_text = response.content if hasattr(response, "content") else str(response)
                    batch_results = self._parse_llm_response(raw_text)
                    all_results.extend(batch_results)
                    break
                except Exception as e:
                    err = str(e)
                    if ("429" in err or "rate_limit" in err or "RESOURCE_EXHAUSTED" in err) and attempt < 2:
                        delay = 60
                        retry_match = re.search(r'retry.*?(\d+)', err, re.IGNORECASE)
                        if retry_match:
                            delay = max(int(retry_match.group(1)), 10)
                        print(f"    ⏳ {label} batch {batch_idx+1}/{n_batches}: rate limited, waiting {delay}s (attempt {attempt+1}/3)")
                        time.sleep(delay)
                    else:
                        print(f"    ⚠️ {label} batch {batch_idx+1}/{n_batches}: {e}")
                        break

        # Final summary line (overwrites progress)
        elapsed = time.time() - extract_start
        print(f"    {label} done: {len(all_results)} found in {n_batches} batches ({elapsed:.1f}s)          ")

        return all_results

    def extract_company_year(
        self,
        company_id: str,
        year: str
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Extract barriers and motivators for a single company-year group.

        Returns two lists of (chunk_id, text) tuples.
        """
        key = (company_id, year)
        chunks = self.grouped_chunks.get(key, [])

        if not chunks:
            return [], []

        company_name = self._get_company_name(company_id)

        # MAP: Extract from this group
        barriers = self._map_extract(
            chunks, company_name, company_id, year, "barriers")
        motivators = self._map_extract(
            chunks, company_name, company_id, year, "motivators")

        return barriers, motivators

    def extract_company_data(self, company_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract barriers and motivators for a company across all years.

        Each extracted item is traced back to its source chunk via chunk_id
        embedded in LLM output, inheriting all BERT metadata.
        """
        company_name = self._get_company_name(company_id)

        print(f"\n{'='*60}")
        print(f"Extracting: {company_name} ({company_id})")
        print(f"{'='*60}")

        years = self.get_years_for_company(company_id)
        batch_size = self.config.batch_size
        total_chunks = sum(len(self.grouped_chunks.get(
            (company_id, y), [])) for y in years)
        total_batches = sum((len(self.grouped_chunks.get(
            (company_id, y), [])) + batch_size - 1) // batch_size for y in years)
        print(f"Years: {years}")
        print(
            f"Chunks: {total_chunks} | Batches: {total_batches} × 2 (barriers+motivators) = {total_batches * 2} LLM calls")

        barrier_rows, motivator_rows = [], []
        total_barriers, total_motivators = 0, 0
        matched_barriers, matched_motivators = 0, 0

        for year in tqdm(years, desc=f"  {company_id}", leave=False):
            key = (company_id, year)
            chunks = self.grouped_chunks.get(key, [])

            # Build chunk_id -> chunk lookup for this group
            chunk_lookup = {
                c.metadata.get("chunk_id"): c
                for c in chunks
                if c.metadata.get("chunk_id")
            }

            barriers, motivators = self.extract_company_year(company_id, year)

            # Create one row per barrier with BERT metadata
            for chunk_id, barrier_text in barriers:
                row = {
                    "company_id": company_id,
                    "company": company_name,
                    "year": year,
                    "barriers": barrier_text,
                }
                # Look up source chunk by chunk_id
                if chunk_id in chunk_lookup:
                    row.update(self._get_chunk_metadata(
                        chunk_lookup[chunk_id]))
                    matched_barriers += 1
                else:
                    print(f"    ⚠️ chunk_id '{chunk_id}' not found in lookup")
                    row["source_chunk_id"] = chunk_id  # Still record the ID
                barrier_rows.append(row)
            total_barriers += len(barriers)

            # Create one row per motivator with BERT metadata
            for chunk_id, motivator_text in motivators:
                row = {
                    "company_id": company_id,
                    "company": company_name,
                    "year": year,
                    "motivators": motivator_text,
                }
                # Look up source chunk by chunk_id
                if chunk_id in chunk_lookup:
                    row.update(self._get_chunk_metadata(
                        chunk_lookup[chunk_id]))
                    matched_motivators += 1
                else:
                    print(f"    ⚠️ chunk_id '{chunk_id}' not found in lookup")
                    row["source_chunk_id"] = chunk_id  # Still record the ID
                motivator_rows.append(row)
            total_motivators += len(motivators)

        # Report match rates
        barrier_match_rate = matched_barriers / \
            total_barriers * 100 if total_barriers > 0 else 0
        motivator_match_rate = matched_motivators / \
            total_motivators * 100 if total_motivators > 0 else 0
        print(
            f"  ✓ Extracted {total_barriers} barriers ({barrier_match_rate:.0f}% ID-matched), "
            f"{total_motivators} motivators ({motivator_match_rate:.0f}% ID-matched)")

        return pd.DataFrame(barrier_rows), pd.DataFrame(motivator_rows)

    def extract_all_companies(self) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Extract barriers and motivators for all companies. Always saves results."""
        # Chunk statistics
        chunk_sizes = [len(c.page_content) for c in self.chunks]
        total_chunks = len(chunk_sizes)
        avg_chunk_chars = sum(
            chunk_sizes) // total_chunks if total_chunks else 0
        chunk_range = (min(chunk_sizes), max(chunk_sizes)
                       ) if chunk_sizes else (0, 0)

        # Estimate LLM calls
        batch_size = self.config.batch_size
        total_batches = sum(
            (len(chunks) + batch_size - 1) // batch_size
            for chunks in self.grouped_chunks.values()
        )
        total_llm_calls = total_batches * 2  # barriers + motivators

        print(f"\n{'='*70}")
        print("EXTRACTION RUN")
        print(f"{'='*70}")
        print(f"  LLM: {self.config.llm_provider}/{self.config.model}")
        print(
            f"  Context: {self.config.ctx:,} tokens → {batch_size} chunks/batch")
        print(
            f"  Chunks: {total_chunks} ({avg_chunk_chars} avg chars, {chunk_range[0]}-{chunk_range[1]} range)")
        print(
            f"  Groups: {len(self.grouped_chunks)} | Est. LLM calls: {total_llm_calls}")
        print(f"{'='*70}\n")

        start_time = time.time()
        results = {}

        for company_id in self.get_companies():
            try:
                df_barriers, df_motivators = self.extract_company_data(
                    company_id)
                results[company_id] = (df_barriers, df_motivators)
                self.save_company_tables(
                    company_id, df_barriers, df_motivators)
            except Exception as e:
                print(f"✗ {company_id}: {e}")

        elapsed = time.time() - start_time
        total_barriers = sum(len(df[0]) for df in results.values())
        total_motivators = sum(len(df[1]) for df in results.values())

        # Save run stats
        run_stats = self._save_stats(
            elapsed=elapsed,
            total_barriers=total_barriers,
            total_motivators=total_motivators,
            total_llm_calls=total_llm_calls,
            total_chunks=total_chunks,
            n_companies=len(results),
        )

        print(f"\n{'='*70}")
        print("✓ EXTRACTION COMPLETE")
        print(f"{'='*70}")
        print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        print(
            f"  Results: {total_barriers} barriers, {total_motivators} motivators")
        print(f"  Saved: {self.config.output_folder}/stats.json")
        print(f"{'='*70}\n")

        return results

    def save_test_run(
        self,
        company_id: str,
        year: str,
        barriers: List[Tuple[str, str]],
        motivators: List[Tuple[str, str]],
        elapsed: float,
        output_folder: str = "../out/test",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Save test run results with same format as full pipeline.

        Args:
            company_id: Company ID tested
            year: Year tested
            barriers: List of (chunk_id, text) tuples from extraction
            motivators: List of (chunk_id, text) tuples from extraction
            elapsed: Time taken in seconds
            output_folder: Where to save results

        Returns:
            (df_barriers, df_motivators) DataFrames with BERT metadata
        """
        company_name = self._get_company_name(company_id)

        # Build chunk lookup for BERT metadata
        key = (company_id, year)
        chunks = self.grouped_chunks.get(key, [])
        chunk_lookup = {c.metadata.get("chunk_id"): c for c in chunks}

        # Build rows with BERT metadata (same as extract_company_data)
        barrier_rows = []
        for chunk_id, text in barriers:
            row = {"company_id": company_id, "company": company_name,
                   "year": year, "barriers": text}
            if chunk_id in chunk_lookup:
                row.update(self._get_chunk_metadata(chunk_lookup[chunk_id]))
            else:
                row["source_chunk_id"] = chunk_id
            barrier_rows.append(row)

        motivator_rows = []
        for chunk_id, text in motivators:
            row = {"company_id": company_id, "company": company_name,
                   "year": year, "motivators": text}
            if chunk_id in chunk_lookup:
                row.update(self._get_chunk_metadata(chunk_lookup[chunk_id]))
            else:
                row["source_chunk_id"] = chunk_id
            motivator_rows.append(row)

        df_barriers = pd.DataFrame(barrier_rows)
        df_motivators = pd.DataFrame(motivator_rows)

        # Save CSVs
        self.save_company_tables(
            company_id, df_barriers, df_motivators, output_folder)

        # Save stats
        self._save_stats(
            elapsed=elapsed,
            total_barriers=len(barriers),
            total_motivators=len(motivators),
            total_llm_calls=2,  # 1 for barriers, 1 for motivators
            total_chunks=len(chunks),
            n_companies=1,
            output_folder=output_folder,
            extra={"test_run": {"company_id": company_id, "year": year}},
        )

        print(f"✓ Test run saved to {output_folder}/")
        return df_barriers, df_motivators

    def _save_stats(
        self,
        elapsed: float,
        total_barriers: int,
        total_motivators: int,
        total_llm_calls: int = 0,
        total_chunks: int = 0,
        n_companies: int = 1,
        output_folder: str = None,
        extra: Dict = None,
    ) -> Dict:
        """Build and save run statistics to stats.json."""
        folder = output_folder or self.config.output_folder

        run_stats = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "config": {
                "llm_provider": self.config.llm_provider,
                "model": self.config.model,
                "ctx": self.config.ctx,
                "batch_size": self.config.batch_size,
                "min_detector_score": self.config.min_detector_score,
            },
            "prompts": {
                "barrier": BARRIER_MAP_PROMPT.messages[0].prompt.template,
                "motivator": MOTIVATOR_MAP_PROMPT.messages[0].prompt.template,
            },
            "results": {
                "barriers": total_barriers,
                "motivators": total_motivators,
                "llm_calls": total_llm_calls,
                "companies": n_companies,
                "chunks_processed": total_chunks,
            },
            "performance": {
                "seconds_per_call": round(elapsed / total_llm_calls, 2) if total_llm_calls else 0,
            },
        }

        if extra:
            run_stats.update(extra)

        os.makedirs(folder, exist_ok=True)
        stats_path = os.path.join(folder, "stats.json")
        with open(stats_path, "w") as f:
            json.dump(run_stats, f, indent=2)

        return run_stats

    def save_company_tables(self, company_id: str, df_barriers: pd.DataFrame, df_motivators: pd.DataFrame, output_folder: str = None):
        """Save extraction results to CSV."""
        folder = output_folder or self.config.output_folder
        os.makedirs(folder, exist_ok=True)

        base_barrier = os.path.join(folder, f"barriers_{company_id}")
        base_motivator = os.path.join(folder, f"motivators_{company_id}")

        df_barriers.to_csv(f"{base_barrier}.csv", index=False)
        df_barriers.to_excel(f"{base_barrier}.xlsx", index=False)
        df_motivators.to_csv(f"{base_motivator}.csv", index=False)
        df_motivators.to_excel(f"{base_motivator}.xlsx", index=False)

    # -------------------------------------------------------------------------
    # Inspection
    # -------------------------------------------------------------------------

    def display_results(self, company_id: str, df_barriers: pd.DataFrame, df_motivators: pd.DataFrame):
        """Display extraction results."""
        print(f"\n{'='*60}")
        print(f"BARRIERS - {company_id} ({len(df_barriers)} items)")
        print(f"{'='*60}")
        if not df_barriers.empty:
            print(tabulate(df_barriers[['year', 'barriers']],
                           headers='keys', tablefmt='grid', maxcolwidths=[None, 60]))
        else:
            print("No barriers found.")

        print(f"\n{'='*60}")
        print(f"MOTIVATORS - {company_id} ({len(df_motivators)} items)")
        print(f"{'='*60}")
        if not df_motivators.empty:
            print(tabulate(df_motivators[['year', 'motivators']],
                           headers='keys', tablefmt='grid', maxcolwidths=[None, 60]))
        else:
            print("No motivators found.")

    def cleanup(self):
        """Release resources."""
        self.gpu.clear()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_pipeline(config: RAGConfig) -> RAGPipeline:
    """Create pipeline and load chunks from cache."""
    pipeline = RAGPipeline(config)
    pipeline.load_from_cache()
    return pipeline
