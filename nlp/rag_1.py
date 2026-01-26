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

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from nlp.data_loader import load_prep_cache, load_bert_cache
from nlp.gpu_utils import GPUManager

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline."""

    # Cache source
    cache_dir: str = "cache"
    use_bert_cache: bool = True  # Default to bert.json (ClimateBERT filtered)

    # Output
    output_folder: str = "out"

    # LLM settings (Ollama)
    ollama_model: str = "llama3.1:8b"
    # phi3:mini
    ollama_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.0
    # Fits ~20 chunks + prompt per LLM call
    llm_num_ctx: int = 8192

    # Extraction settings
    max_chunks_per_group: int = 25  # Safety limit per company-year group


# =============================================================================
# PROMPTS
# =============================================================================

# MAP phase: Extract from a single company-year group
# basically using the LLM as a semantic filter + high-recall extractor, not a generator
# scopeREQ: The sentence MUST explicitly mention at least one emissions-related term (e.g. "emissions", "CO₂", "GHG", "carbon", "decarbonisation", "net zero", "carbon neutral").
BARRIER_MAP_PROMPT = ChatPromptTemplate.from_template("""
You are analyzing sustainability report excerpts from {company} ({year}) to extract BARRIERS to DECARBONISATION.

TASK:
Extract ALL barriers that explicitly hinder, slow, limit, or complicate efforts to reduce greenhouse gas (GHG) emissions or to decarbonise operations, products, or value chains.

DEFINITION:
A "barrier" is any stated challenge, constraint, limitation, obstacle, risk, dependency, or external factor that makes decarbonisation or GHG emissions reduction more difficult, slower, more expensive, or more uncertain.

CHUNK ANNOTATIONS:
- Chunks marked [NET-ZERO] contain net-zero/reduction targets
- Chunks marked [COMMITMENT] contain firm commitments
- Chunks marked [SPECIFIC] contain specific/quantified claims
- Prioritize extracting from marked chunks, but also check unmarked chunks

SCOPE REQUIREMENT (MANDATORY):
To qualify, the barrier MUST be explicitly linked in the text to at least one of the following:
- Decarbonisation
- Reduction of CO₂ or other greenhouse gas emissions
- Net-zero, carbon neutrality, or emissions targets
- Transition away from fossil fuels or high-carbon activities

DO NOT include barriers that relate only to:
- General sustainability or ESG topics
- Biodiversity, water, waste, safety, or social issues
- Business or operational challenges unless they are explicitly tied to emissions reduction or decarbonisation

RULES:
- Extract EVERY qualifying barrier mentioned; do NOT filter, rank, or prioritize
- Include barriers even if mentioned briefly or indirectly, as long as the link to decarbonisation or emissions reduction is explicit
- Do NOT infer or assume a link to decarbonisation if it is not stated in the text
- Return ONLY verbatim text copied exactly from the report (no paraphrasing, no rewriting)
- Do NOT normalize capitalization, punctuation, spacing, or units.
- Each output must be the full original sentence or bullet text that contains the barrier
- If a single sentence contains multiple qualifying barriers, include the same sentence multiple times (once per barrier)
- Evaluate each sentence independently. Do NOT combine information across sentences or infer missing context from earlier or later text
- If no qualifying barriers are found, respond with exactly: NONE_FOUND

REPORT EXCERPTS:
{context}

OUTPUT FORMAT:
- One line per extracted sentence
- Each line must start with "- "
- No empty lines
- No commentary or headers

Extract ALL qualifying barriers:
""")

MOTIVATOR_MAP_PROMPT = ChatPromptTemplate.from_template("""
You are analyzing sustainability report excerpts from {company} ({year}) to extract MOTIVATORS for DECARBONISATION.

TASK:
Extract ALL motivators that explicitly drive, encourage, justify, or accelerate efforts to reduce greenhouse gas (GHG) emissions or to decarbonise operations, products, or value chains.

DEFINITION:
A "motivator" is any stated driver, incentive, benefit, opportunity, pressure, commitment, obligation, or strategic reason that encourages or justifies decarbonisation or GHG emissions reduction.

CHUNK ANNOTATIONS:
- Chunks marked [NET-ZERO] contain net-zero/reduction targets
- Chunks marked [COMMITMENT] contain firm commitments
- Chunks marked [SPECIFIC] contain specific/quantified claims
- Prioritize extracting from marked chunks, but also check unmarked chunks

SCOPE REQUIREMENT (MANDATORY):
To qualify, the motivator MUST be explicitly linked in the text to at least one of the following:
- Decarbonisation
- Reduction of CO₂ or other greenhouse gas emissions
- Net-zero, carbon neutrality, or emissions targets
- Transition away from fossil fuels or high-carbon activities

DO NOT include motivators that relate only to:
- General sustainability or ESG ambitions
- Biodiversity, water, waste, safety, or social performance
- Business strategies or opportunities unless they are explicitly connected to emissions reduction or decarbonisation

RULES:
- Extract EVERY qualifying motivator mentioned; do NOT filter, rank, or prioritize
- Include motivators even if mentioned briefly or indirectly, as long as the link to decarbonisation or emissions reduction is explicit
- Do NOT infer or assume a link to decarbonisation if it is not stated in the text
- Return ONLY verbatim text copied exactly from the report (no paraphrasing, no rewriting)
- Do NOT normalize capitalization, punctuation, spacing, or units.
- Each output must be the full original sentence or bullet text that contains the motivator
- If a single sentence contains multiple qualifying motivators, include the same sentence multiple times (once per motivator)
- Evaluate each sentence independently. Do NOT combine information across sentences or infer missing context from earlier or later text
- If no qualifying motivators are found, respond with exactly: NONE_FOUND

REPORT EXCERPTS:
{context}

Extract ALL qualifying motivators (one per line, starting with "- "):
""")


# =============================================================================
# MAIN PIPELINE CLASS
# =============================================================================

class RAGPipeline:
    """RAG pipeline for extracting barriers and motivators from sustainability reports."""

    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize the pipeline."""
        self.config = config or RAGConfig()
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
        """Lazy-load the Ollama LLM."""
        if self._llm is None:
            print(f"Loading Ollama model: {self.config.ollama_model}")
            self._llm = ChatOllama(
                model=self.config.ollama_model,
                base_url=self.config.ollama_base_url,
                temperature=self.config.llm_temperature,
                num_ctx=self.config.llm_num_ctx,
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

    def _find_source_chunk_exact(self, extracted_text: str, chunks: List) -> Optional[Dict[str, Any]]:
        """Find source chunk using exact substring match.

        Since prompts require verbatim extraction, we can use exact string matching
        to trace extracted sentences back to source chunks.

        Returns dict with source chunk ID and all BERT metadata, or None if not found.
        """
        if not extracted_text or not chunks:
            return None

        extracted_clean = extracted_text.strip().lower()

        for chunk in chunks:
            chunk_text = chunk.page_content.lower() if hasattr(chunk, 'page_content') else str(chunk).lower()
            if extracted_clean in chunk_text:
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
        return None

    def _format_chunk_with_metadata(self, i: int, chunk) -> str:
        """Format chunk with BERT labels for prompt context.

        Annotates chunks with labels like [NET-ZERO], [COMMITMENT], [SPECIFIC]
        to help the LLM prioritize high-quality chunks.
        """
        labels = []
        if chunk.metadata.get("netzero_label") == "reduction":
            labels.append("NET-ZERO")
        if chunk.metadata.get("commitment_label") == "yes":
            labels.append("COMMITMENT")
        if chunk.metadata.get("specificity_label") == "specific":
            labels.append("SPECIFIC")

        tag = f" [{', '.join(labels)}]" if labels else ""
        return f"[Chunk {i+1}{tag}]\n{chunk.page_content}"

    def _sort_chunks_by_quality(self, chunks: List) -> List:
        """Sort chunks by composite quality score (best first).

        Score = 30% detector + 35% specificity + 35% commitment
        """
        def score(c):
            return (
                c.metadata.get("detector_score", 0) * 0.3 +
                c.metadata.get("specificity_score", 0) * 0.35 +
                c.metadata.get("commitment_score", 0) * 0.35
            )
        return sorted(chunks, key=score, reverse=True)

    def _compute_group_metadata(self, chunks: List) -> Dict[str, Any]:
        """Compute aggregate BERT metadata for a group of chunks.

        Used as fallback when exact substring match fails (e.g., LLM slightly modified text).
        Returns mean scores and most common labels.
        """
        if not chunks:
            return {
                "source_chunk_id": None,
                "detector_score": 0,
                "specificity_score": 0,
                "specificity_label": None,
                "commitment_score": 0,
                "commitment_label": None,
                "sentiment_label": None,
                "sentiment_score": 0,
                "netzero_label": None,
                "netzero_score": 0,
            }

        # Compute mean scores
        detector_scores = [c.metadata.get("detector_score", 0) for c in chunks]
        specificity_scores = [c.metadata.get("specificity_score", 0) for c in chunks]
        commitment_scores = [c.metadata.get("commitment_score", 0) for c in chunks]
        sentiment_scores = [c.metadata.get("sentiment_score", 0) for c in chunks]
        netzero_scores = [c.metadata.get("netzero_score", 0) for c in chunks]

        # Get most common labels (mode)
        def mode(values: List) -> Optional[str]:
            filtered = [v for v in values if v is not None]
            if not filtered:
                return None
            from collections import Counter
            return Counter(filtered).most_common(1)[0][0]

        specificity_labels = [c.metadata.get("specificity_label") for c in chunks]
        commitment_labels = [c.metadata.get("commitment_label") for c in chunks]
        sentiment_labels = [c.metadata.get("sentiment_label") for c in chunks]
        netzero_labels = [c.metadata.get("netzero_label") for c in chunks]

        return {
            "source_chunk_id": None,  # No exact match
            "detector_score": float(np.mean(detector_scores)) if detector_scores else 0,
            "specificity_score": float(np.mean(specificity_scores)) if specificity_scores else 0,
            "specificity_label": mode(specificity_labels),
            "commitment_score": float(np.mean(commitment_scores)) if commitment_scores else 0,
            "commitment_label": mode(commitment_labels),
            "sentiment_label": mode(sentiment_labels),
            "sentiment_score": float(np.mean(sentiment_scores)) if sentiment_scores else 0,
            "netzero_label": mode(netzero_labels),
            "netzero_score": float(np.mean(netzero_scores)) if netzero_scores else 0,
        }

    def _parse_llm_response(self, response_text: str) -> List[str]:
        """Parse LLM response into list of items."""
        if not response_text:
            return []

        text = response_text.strip()
        if "NONE_FOUND" in text.upper():
            return []

        items = []
        seen: Set[str] = set()

        for line in text.splitlines():
            line = line.strip()
            if not line or len(line) < 10:
                continue

            # Remove bullet points
            if line.startswith(("-", "•", "*", "–", "·")):
                line = line[1:].strip()
            # Remove numbering
            if len(line) > 2 and line[0].isdigit() and line[1] in ".):":
                line = line[2:].strip()

            if not line:
                continue

            normalized = line.lower().strip()
            if normalized in seen:
                continue

            # Fuzzy deduplication: check word overlap
            is_duplicate = False
            for existing in seen:
                existing_words = set(existing.split())
                line_words = set(normalized.split())
                if len(existing_words) > 3 and len(line_words) > 3:
                    overlap = len(existing_words & line_words)
                    min_len = min(len(existing_words), len(line_words))
                    if overlap >= min_len * 0.7:
                        is_duplicate = True
                        break

            if not is_duplicate:
                items.append(line)
                seen.add(normalized)

        return items

    def _map_extract(
        self,
        chunks: List,
        company: str,
        year: str,
        extract_type: str
    ) -> List[str]:
        """MAP phase: Extract barriers/motivators from a single company-year group.

        Uses BERT metadata to:
        1. Sort chunks by quality (best first)
        2. Annotate chunks with BERT labels in the prompt
        """
        if not chunks:
            return []

        # Sort by BERT quality score (best chunks first)
        chunks = self._sort_chunks_by_quality(chunks)

        # Limit chunks if too many
        if len(chunks) > self.config.max_chunks_per_group:
            chunks = chunks[:self.config.max_chunks_per_group]

        # Build context WITH metadata annotations
        context = "\n\n---\n\n".join(
            self._format_chunk_with_metadata(i, chunk)
            for i, chunk in enumerate(chunks)
        )

        # Select prompt
        prompt = BARRIER_MAP_PROMPT if extract_type == "barriers" else MOTIVATOR_MAP_PROMPT
        chain = prompt | self.llm

        try:
            response = chain.invoke({
                "company": company,
                "year": year,
                "context": context
            })
            raw_text = response.content if hasattr(
                response, "content") else str(response)
            return self._parse_llm_response(raw_text)
        except Exception as e:
            print(f"    ⚠️ Error in map phase: {e}")
            return []

    def extract_company_year(
        self,
        company_id: str,
        year: str
    ) -> Tuple[List[str], List[str]]:
        """Extract barriers and motivators for a single company-year group."""
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

        Each extracted item is traced back to its source chunk via exact substring
        matching, inheriting all BERT metadata (detector, specificity, commitment,
        sentiment, netzero scores and labels).
        """
        company_name = self._get_company_name(company_id)

        print(f"\n{'='*60}")
        print(f"Extracting: {company_name} ({company_id})")
        print(f"{'='*60}")

        years = self.get_years_for_company(company_id)
        print(f"Years: {years}")
        print(
            f"Groups: {len(years)} | Avg chunks/group: {len(self.chunks) / max(len(self.grouped_chunks), 1):.1f}")

        barrier_rows, motivator_rows = [], []
        total_barriers, total_motivators = 0, 0
        matched_barriers, matched_motivators = 0, 0

        for year in tqdm(years, desc=f"  {company_id}", leave=False):
            key = (company_id, year)
            chunks = self.grouped_chunks.get(key, [])

            barriers, motivators = self.extract_company_year(company_id, year)

            # Create one row per barrier with BERT metadata
            for barrier in barriers:
                row = {
                    "company_id": company_id,
                    "company": company_name,
                    "year": year,
                    "barriers": barrier,
                }
                # Find source chunk and inherit BERT scores
                source = self._find_source_chunk_exact(barrier, chunks)
                if source:
                    row.update(source)
                    matched_barriers += 1
                else:
                    # Fallback: use group-level aggregates
                    row.update(self._compute_group_metadata(chunks))
                barrier_rows.append(row)
            total_barriers += len(barriers)

            # Create one row per motivator with BERT metadata
            for motivator in motivators:
                row = {
                    "company_id": company_id,
                    "company": company_name,
                    "year": year,
                    "motivators": motivator,
                }
                # Find source chunk and inherit BERT scores
                source = self._find_source_chunk_exact(motivator, chunks)
                if source:
                    row.update(source)
                    matched_motivators += 1
                else:
                    # Fallback: use group-level aggregates
                    row.update(self._compute_group_metadata(chunks))
                motivator_rows.append(row)
            total_motivators += len(motivators)

        # Report match rates
        barrier_match_rate = matched_barriers / total_barriers * 100 if total_barriers > 0 else 0
        motivator_match_rate = matched_motivators / total_motivators * 100 if total_motivators > 0 else 0
        print(
            f"  ✓ Extracted {total_barriers} barriers ({barrier_match_rate:.0f}% matched), "
            f"{total_motivators} motivators ({motivator_match_rate:.0f}% matched)")

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
            "LLM": "✓" if self._llm else "○ (lazy)",
        }

        for k, v in status.items():
            print(f"  {k}: {v}")

        print(f"\nConfig:")
        print(f"  Cache: {self.config.cache_dir}")
        print(
            f"  Source: {'bert.json' if self.config.use_bert_cache else 'prep.json'}")
        print(f"  Ollama: {self.config.ollama_model}")
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

def quick_start(
    cache_dir: str = "cache",
    use_bert_cache: bool = True,
    ollama_model: str = "llama3.1:8b"
) -> RAGPipeline:
    """Quick start: load cache and prepare for extraction."""
    config = RAGConfig(
        cache_dir=cache_dir,
        use_bert_cache=use_bert_cache,
        ollama_model=ollama_model,
    )
    pipeline = RAGPipeline(config)
    pipeline.load_from_cache()
    return pipeline


def extract_all(
    cache_dir: str = "cache",
    output_folder: str = "out",
    ollama_model: str = "llama3.1:8b"
) -> Dict:
    """Run full extraction pipeline."""
    config = RAGConfig(
        cache_dir=cache_dir,
        output_folder=output_folder,
        ollama_model=ollama_model,
        use_bert_cache=True,
    )
    pipeline = RAGPipeline(config)
    pipeline.load_from_cache()
    pipeline.print_chunk_overview()
    return pipeline.extract_all_companies()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Quick test
    pipeline = RAGPipeline()
    pipeline.load_from_cache()
    pipeline.print_status()

    # Uncomment to run extraction:
    pipeline.extract_all_companies()
