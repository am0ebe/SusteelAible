"""
RAG Pipeline for Company Decarbonisation Report Analysis
========================================================

This module provides a pipeline for:
1. Loading preprocessed data from JSON cache files (bert.json with ClimateBERT filtering)
2. Extracting barriers and motivators using map-reduce LLM strategy
3. Aggregating results by company and year

Strategy: ClimateBERT filters → Group by company-year → Map (extract per group) → Reduce (aggregate)

Usage:
    from rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()
    pipeline.load_from_cache()
    pipeline.extract_all_companies()
"""

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set

import pandas as pd
from tqdm import tqdm
from tabulate import tabulate

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from nlp.json_loader import load_prep_cache, load_bert_cache
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
    # Increased for map-reduce (fits ~20 chunks + prompt)
    llm_num_ctx: int = 8192

    # Extraction settings
    max_chunks_per_group: int = 25  # Safety limit per company-year group


# =============================================================================
# PROMPTS
# =============================================================================

# MAP phase: Extract from a single company-year group
BARRIER_MAP_PROMPT = ChatPromptTemplate.from_template("""You are analyzing sustainability report excerpts from {company} ({year}) to extract BARRIERS to decarbonisation.

TASK: Extract ALL specific barriers mentioned in the text below. Capture every barrier, even minor ones.

RULES:
1. Extract EVERY barrier mentioned - do not filter or prioritize
2. Each barrier should be a concrete, specific challenge
3. Include the barrier even if mentioned briefly or indirectly
4. Use concise phrasing (5-20 words per barrier)
5. If truly no barriers found, respond with: NONE_FOUND

REPORT EXCERPTS:
{context}

Extract ALL barriers (one per line, starting with "- "):
""")

MOTIVATOR_MAP_PROMPT = ChatPromptTemplate.from_template("""You are analyzing sustainability report excerpts from {company} ({year}) to extract MOTIVATORS/DRIVERS for decarbonisation.

TASK: Extract ALL specific motivators mentioned in the text below. Capture every driver, even minor ones.

RULES:
1. Extract EVERY motivator mentioned - do not filter or prioritize
2. Each motivator should be a concrete, specific driver
3. Include the motivator even if mentioned briefly or indirectly
4. Use concise phrasing (5-20 words per motivator)
5. If truly no motivators found, respond with: NONE_FOUND

REPORT EXCERPTS:
{context}

Extract ALL motivators (one per line, starting with "- "):
""")

# REDUCE phase: Aggregate and deduplicate across chunks (if needed for very large groups)
REDUCE_PROMPT = ChatPromptTemplate.from_template("""You are consolidating extracted {extract_type} from multiple report sections.

TASK: Merge and deduplicate the following {extract_type}, preserving all unique items.

RULES:
1. Keep ALL unique {extract_type} - do not remove any
2. Merge duplicates that say the same thing differently
3. Preserve specific details and numbers
4. Output one item per line, starting with "- "

EXTRACTED {extract_type}:
{items}

Consolidated {extract_type} (one per line, starting with "- "):
""")


# =============================================================================
# MAIN PIPELINE CLASS
# =============================================================================

class RAGPipeline:
    """RAG pipeline using map-reduce extraction strategy."""

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
        """Group chunks by (company_id, year) for map-reduce processing."""
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
    # Extraction (Map-Reduce)
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
        """MAP phase: Extract barriers/motivators from a single company-year group."""
        if not chunks:
            return []

        # Limit chunks if too many
        if len(chunks) > self.config.max_chunks_per_group:
            chunks = chunks[:self.config.max_chunks_per_group]

        # Build context from all chunks
        context = "\n\n---\n\n".join(
            f"[Chunk {i+1}]\n{chunk.page_content}"
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

    def _reduce_items(self, all_items: List[str], extract_type: str) -> List[str]:
        """REDUCE phase: Consolidate items if list is very long."""
        if len(all_items) <= 30:
            # No need to reduce, just deduplicate
            seen = set()
            unique = []
            for item in all_items:
                normalized = item.lower().strip()
                if normalized not in seen:
                    seen.add(normalized)
                    unique.append(item)
            return unique

        # Use LLM to consolidate
        chain = REDUCE_PROMPT | self.llm
        items_text = "\n".join(f"- {item}" for item in all_items)

        try:
            response = chain.invoke({
                "extract_type": extract_type,
                "items": items_text
            })
            raw_text = response.content if hasattr(
                response, "content") else str(response)
            return self._parse_llm_response(raw_text)
        except Exception as e:
            print(f"    ⚠️ Error in reduce phase: {e}")
            return all_items[:30]  # Fallback: return first 30

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
        """Extract barriers and motivators for a company across all years."""
        company_name = self._get_company_name(company_id)

        print(f"\n{'='*60}")
        print(f"Extracting: {company_name} ({company_id})")
        print(f"{'='*60}")

        years = self.get_years_for_company(company_id)
        print(f"Years: {years}")
        print(
            f"Groups: {len(years)} | Avg chunks/group: {len(self.chunks) / max(len(self.grouped_chunks), 1):.1f}")

        barrier_rows, motivator_rows = [], []

        for year in tqdm(years, desc=f"  {company_id}", leave=False):
            key = (company_id, year)
            n_chunks = len(self.grouped_chunks.get(key, []))

            barriers, motivators = self.extract_company_year(company_id, year)

            barrier_rows.append({
                "company_id": company_id,
                "company": company_name,
                "year": year,
                "chunks": n_chunks,
                "barriers": "\n".join(barriers) if barriers else "NONE_FOUND",
                "barrier_count": len(barriers),
            })
            motivator_rows.append({
                "company_id": company_id,
                "company": company_name,
                "year": year,
                "chunks": n_chunks,
                "motivators": "\n".join(motivators) if motivators else "NONE_FOUND",
                "motivator_count": len(motivators),
            })

        print(f"  ✓ Extracted {sum(r['barrier_count'] for r in barrier_rows)} barriers, "
              f"{sum(r['motivator_count'] for r in motivator_rows)} motivators")

        return pd.DataFrame(barrier_rows), pd.DataFrame(motivator_rows)

    def extract_all_companies(self, save_results: bool = True) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Extract barriers and motivators for all companies."""
        print(f"\n{'='*70}")
        print("EXTRACTING ALL COMPANIES (Map-Reduce Strategy)")
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
        total_barriers = sum(
            df[0]['barrier_count'].sum() for df in results.values()
        )
        total_motivators = sum(
            df[1]['motivator_count'].sum() for df in results.values()
        )

        print(f"\n{'='*70}")
        print("✓ EXTRACTION COMPLETE")
        print(f"{'='*70}")
        print(f"  Companies: {len(results)}")
        print(f"  Total barriers: {total_barriers}")
        print(f"  Total motivators: {total_motivators}")
        print(f"{'='*70}\n")

        return results

    def save_company_tables(self, company_id: str, df_barriers: pd.DataFrame, df_motivators: pd.DataFrame):
        """Save extraction results to CSV and Excel."""
        os.makedirs(self.config.output_folder, exist_ok=True)

        base_barrier = os.path.join(
            self.config.output_folder, f"barriers_{company_id}")
        base_motivator = os.path.join(
            self.config.output_folder, f"motivators_{company_id}")

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
        print(f"BARRIERS - {company_id}")
        print(f"{'='*60}")
        print(tabulate(df_barriers[['year', 'chunks', 'barrier_count', 'barriers']],
                       headers='keys', tablefmt='grid', maxcolwidths=[None, None, None, 60]))

        print(f"\n{'='*60}")
        print(f"MOTIVATORS - {company_id}")
        print(f"{'='*60}")
        print(tabulate(df_motivators[['year', 'chunks', 'motivator_count', 'motivators']],
                       headers='keys', tablefmt='grid', maxcolwidths=[None, None, None, 60]))

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
