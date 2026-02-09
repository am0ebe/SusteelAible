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

from nlp.gpu_utils import GPUManager
from nlp.data_loader import load_prep_cache, load_bert_cache
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Literal

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

    # !! Context window: Ollama uses for actual context, Groq uses for batch size calc only
    llm_num_ctx: int = 8096

    # Stop sequences to halt generation early
    # Extensions: "Note:", "Explanation:", "Here are", "I found"
    stop_tokens: tuple = ("NONE_FOUND\n", "\n\n\n")

    # Pipeline
    cache_dir: str = "../cache"
    use_bert_cache: bool = True
    output_folder: str = "../out"
    max_chunks_per_group: int = 7
    min_detector_score: float = 0.7


# =============================================================================
# PROMPTS
# =============================================================================

BARRIER_MAP_PROMPT = ChatPromptTemplate.from_template("""Extract BARRIERS to decarbonisation from {company} ({year}) report.

BARRIER = challenge, constraint, risk, or factor that makes reducing GHG emissions harder.

RULES:
- Copy text EXACTLY as written (verbatim)
- Each chunk starts with [chunk_id] - use that exact ID

OUTPUT FORMAT (STRICT):
[chunk_id]|||verbatim text

EXAMPLE:
[012_003]|||The high cost of green hydrogen limits our decarbonisation options.
[012_007]|||Carbon capture technology remains expensive and unproven at scale.

One per line. No other text. No JSON. No explanations. No quotes.
If none found: NONE_FOUND

TEXT:
{context}""")

MOTIVATOR_MAP_PROMPT = ChatPromptTemplate.from_template("""Extract MOTIVATORS for decarbonisation from {company} ({year}) report.

MOTIVATOR = driver, incentive, commitment, or pressure that encourages reducing GHG emissions.

RULES:
- Copy text EXACTLY as written (verbatim)
- Each chunk starts with [chunk_id] - use that exact ID

OUTPUT FORMAT (STRICT):
[chunk_id]|||verbatim sentence

EXAMPLE:
[012_003]|||EU regulations require 55% emissions reduction by 2030.
[012_007]|||Customer demand for green steel is driving our net-zero commitment.

One per line. No other text. No JSON. No explanations. No quotes.
If none found: NONE_FOUND

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
        """Lazy-load the LLM (Ollama or Groq based on config)."""
        if self._llm is None:
            if self.config.llm_provider == "groq":
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError(
                        "GROQ_API_KEY not found in environment. Check .env file.")
                print(f"Loading Groq model: {self.config.model}")
                self._llm = ChatGroq(
                    model=self.config.model,
                    api_key=api_key,
                    temperature=self.config.llm_temperature,
                    stop_sequences=list(self.config.stop_tokens),
                )
            else:
                print(f"Loading Ollama model: {self.config.model}")
                self._llm = ChatOllama(
                    model=self.config.model,
                    base_url=self.config.ollama_base_url,
                    temperature=self.config.llm_temperature,
                    num_ctx=self.config.llm_num_ctx,
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

        for key, chunks in sorted(self.grouped_chunks.items(), key=lambda x: -len(x[1]))[:10]:
            print(f"{key}: {len(chunks)} chunks")

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
        """Parse text response: [chunk_id]|||verbatim text

        Returns list of (chunk_id, text) tuples.
        """
        if not response_text or "NONE_FOUND" in response_text.upper():
            return []

        # Pattern: [chunk_id]|||text
        pattern = re.compile(r'^\[([^\]]+)\]\|\|\|(.+)$', re.MULTILINE)

        results = []
        seen = set()

        for match in pattern.finditer(response_text):
            chunk_id = match.group(1).strip()
            text = match.group(2).strip()
            # Deduplicate by text
            if text.lower() not in seen:
                results.append((chunk_id, text))
                seen.add(text.lower())

        return results

    def _map_extract(
        self,
        chunks: List,
        company: str,
        year: str,
        extract_type: str
    ) -> List[Tuple[str, str]]:
        """MAP phase: Extract barriers/motivators from a single company-year group.

        Returns list of (chunk_id, extracted_text) tuples.
        Processes ALL chunks in batches of max_chunks_per_group for high recall.
        """
        if not chunks:
            return []

        # Sort by BERT quality score (best chunks first)
        chunks = self._sort_chunks_by_quality(chunks)

        # Select prompt
        prompt = BARRIER_MAP_PROMPT if extract_type == "barriers" else MOTIVATOR_MAP_PROMPT
        chain = prompt | self.llm

        batch_size = self.config.max_chunks_per_group
        n_batches = (len(chunks) + batch_size - 1) // batch_size
        all_results = []

        # Process all chunks in batches
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(chunks))
            batch = chunks[start:end]

            # Build context with chunk IDs (no BERT labels - looked up after)
            context = "\n\n---\n\n".join(
                self._format_chunk(chunk)
                for chunk in batch
            )

            try:
                # DEBUG: Show context being sent (first batch only)
                if batch_idx == 0:
                    print(
                        f"    [DEBUG] Context preview ({len(context)} chars):")
                    print(f"    {context[:500]}...")

                response = chain.invoke({
                    "company": company,
                    "year": year,
                    "context": context
                })
                raw_text = response.content if hasattr(
                    response, "content") else str(response)

                # DEBUG: Show raw LLM output for first batch
                if batch_idx == 0:
                    print(
                        f"    [DEBUG] Raw LLM output ({len(raw_text)} chars):")
                    print(f"    {raw_text[:300]}...")

                batch_results = self._parse_llm_response(raw_text)
                all_results.extend(batch_results)
            except Exception as e:
                print(f"    ⚠️ Error in batch {batch_idx+1}/{n_batches}: {e}")

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
        barriers = self._map_extract(chunks, company_name, year, "barriers")
        motivators = self._map_extract(
            chunks, company_name, year, "motivators")

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
        batch_size = self.config.max_chunks_per_group
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

    def extract_all_companies(self, save_results: bool = True) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Extract barriers and motivators for all companies."""
        print(f"\n{'='*70}")
        print("EXTRACTING ALL COMPANIES")
        print(f"{'='*70}")
        print(f"Total groups: {len(self.grouped_chunks)}")
        print(f"LLM context: {self.config.llm_num_ctx} tokens")

        results = {}
        companies = self.get_companies()

        for company_id in companies:
            try:
                df_barriers, df_motivators = self.extract_company_data(
                    company_id)
                results[company_id] = (df_barriers, df_motivators)

                if save_results:
                    self.save_company_tables(
                        company_id, df_barriers, df_motivators)

            except Exception as e:
                print(f"✗ {company_id}: {e}")

        # Summary
        total_barriers = sum(len(df[0]) for df in results.values())
        total_motivators = sum(len(df[1]) for df in results.values())

        print(f"\n{'='*70}")
        print("✓ EXTRACTION COMPLETE")
        print(f"{'='*70}")
        print(f"  Companies: {len(results)}")
        print(f"  Total barriers: {total_barriers}")
        print(f"  Total motivators: {total_motivators}")
        print(f"{'='*70}\n")

        return results

    def save_company_tables(self, company_id: str, df_barriers: pd.DataFrame, df_motivators: pd.DataFrame):
        """Save extraction results to CSV and Excel.

        Each barrier/motivator is already its own row with metadata preserved.
        """
        os.makedirs(self.config.output_folder, exist_ok=True)

        base_barrier = os.path.join(
            self.config.output_folder, f"barriers_{company_id}")
        base_motivator = os.path.join(
            self.config.output_folder, f"motivators_{company_id}")

        # Save directly (already one row per item)
        df_barriers.to_csv(f"{base_barrier}.csv", index=False)
        df_barriers.to_excel(f"{base_barrier}.xlsx", index=False)
        df_motivators.to_csv(f"{base_motivator}.csv", index=False)
        df_motivators.to_excel(f"{base_motivator}.xlsx", index=False)

    # -------------------------------------------------------------------------
    # Inspection
    # -------------------------------------------------------------------------

    def print_status(self):
        """Show pipeline status."""
        print(f"\n{'='*60}")
        print("PIPELINE STATUS")
        print(f"{'='*60}")

        status = {
            "Chunks loaded": len(self.chunks) if self.chunks else 0,
            "Company-year groups": len(self.grouped_chunks),
            "LLM": "✓" if self._llm else "🐗 (lazy)",
        }

        for k, v in status.items():
            print(f"  {k}: {v}")

        print(f"\nConfig:")
        print(f"  Cache: {self.config.cache_dir}")
        print(
            f"  Source: {'bert.json' if self.config.use_bert_cache else 'prep.json'}")
        if self.config.llm_provider == "groq":
            print(f"  LLM: Groq API ({self.config.model})")
        else:
            print(f"  LLM: Ollama ({self.config.model})")
        print(f"  Context: {self.config.llm_num_ctx} tokens")

        # Print chunk matrix
        if self.chunks:
            self.print_chunk_overview()

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


def extract_all(config: RAGConfig) -> Dict:
    """Run full extraction pipeline."""
    pipeline = load_pipeline(config)
    pipeline.print_chunk_overview()
    return pipeline.extract_all_companies()
