"""
RAG Pipeline for Company Decarbonisation Report Analysis
========================================================

This module provides a pipeline for:
1. Loading preprocessed data from JSON cache files (prep.json or bert.json)
2. Creating/updating FAISS vector stores
3. Extracting barriers and motivators using local LLM (Ollama)

Usage:
    from rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()
    pipeline.load_from_cache()
    pipeline.extract_all_companies()
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set

import pandas as pd
from tqdm import tqdm
from tabulate import tabulate

from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_ollama import ChatOllama
from transformers import AutoTokenizer

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
    use_bert_cache: bool = False  # If True, load bert.json; else prep.json

    # Vector store
    vector_db_base: str = "vector_db"
    vector_db_name: str = "reports"

    # Output
    output_folder: str = "decarbonisation_tables"

    # Embedding model
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    max_tokens: int = 512

    # LLM settings (Ollama)
    ollama_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.0
    llm_num_ctx: int = 4096

    # Retrieval
    retrieval_k: int = 50
    context_chunks: int = 5

    @property
    def vector_db_path(self) -> str:
        return os.path.join(self.vector_db_base, self.vector_db_name)


# =============================================================================
# PROMPTS
# =============================================================================

BARRIER_PROMPT = ChatPromptTemplate.from_template("""You are analyzing company sustainability reports to extract BARRIERS to decarbonisation.

TASK: Extract specific barriers mentioned in the text below. Only extract barriers that are EXPLICITLY stated or clearly implied.

RULES:
1. Each barrier must be a specific, concrete challenge (not vague statements)
2. Only include barriers actually mentioned in the context - DO NOT invent or assume barriers
3. If a barrier is mentioned multiple times, include it only once
4. Use concise phrasing (5-15 words per barrier)
5. If no clear barriers are found, respond with exactly: NONE_FOUND

CONTEXT:
{context}

Extract barriers (one per line, starting with "- "). If none found, write NONE_FOUND:
""")

MOTIVATOR_PROMPT = ChatPromptTemplate.from_template("""You are analyzing company sustainability reports to extract MOTIVATORS for decarbonisation.

TASK: Extract specific motivators/drivers mentioned in the text below. Only extract motivators that are EXPLICITLY stated or clearly implied.

RULES:
1. Each motivator must be a specific, concrete driver (not vague statements)
2. Only include motivators actually mentioned in the context - DO NOT invent or assume motivators
3. If a motivator is mentioned multiple times, include it only once
4. Use concise phrasing (5-15 words per motivator)
5. If no clear motivators are found, respond with exactly: NONE_FOUND

CONTEXT:
{context}

Extract motivators (one per line, starting with "- "). If none found, write NONE_FOUND:
""")

# Semantic search queries
BARRIER_QUERY = (
    "barriers challenges obstacles difficulties problems constraints limitations "
    "impediments hurdles risks issues preventing hindering decarbonization carbon reduction"
)
MOTIVATOR_QUERY = (
    "motivators drivers incentives benefits opportunities reasons goals targets "
    "commitments advantages value business case for decarbonization sustainability"
)


# =============================================================================
# MAIN PIPELINE CLASS
# =============================================================================

class RAGPipeline:
    """RAG pipeline for company decarbonisation report analysis."""

    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize the pipeline."""
        self.config = config or RAGConfig()
        self.gpu = GPUManager()

        # Lazy-loaded components
        self._llm = None
        self._embedding_model = None
        self._tokenizer = None

        # Data
        self.chunks: List = []
        self.vectorstore: Optional[FAISS] = None
        self.retriever = None

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

    @property
    def embedding_model(self):
        """Lazy-load the embedding model."""
        if self._embedding_model is None:
            self._embedding_model = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model_name,
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embedding_model

    @property
    def tokenizer(self):
        """Lazy-load the tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.embedding_model_name
            )
        return self._tokenizer

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------

    def load_from_cache(self) -> List:
        """Load documents from JSON cache files."""
        print(f"\n{'='*60}")
        print("LOADING FROM CACHE")
        print(f"{'='*60}")
        print(f"Cache dir: {self.config.cache_dir}")
        print(
            f"Source: {'bert.json' if self.config.use_bert_cache else 'prep.json'}")

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

        companies = set(c.metadata.get("company_id")
                        for c in self.chunks if c.metadata.get("company_id"))
        years = set(c.metadata.get("year")
                    for c in self.chunks if c.metadata.get("year"))

        print(f"\n✓ Loaded {len(self.chunks)} chunks")
        print(f"  Companies: {len(companies)}")
        print(f"  Years: {sorted(years)}")

        return self.chunks

    # -------------------------------------------------------------------------
    # Token Management
    # -------------------------------------------------------------------------

    def truncate_to_max_tokens(self, text: str) -> Tuple[str, bool]:
        """Truncate text to max tokens if needed."""
        encoded = self.tokenizer(
            text, add_special_tokens=False, truncation=False)
        n_tokens = len(encoded["input_ids"])

        if n_tokens <= self.config.max_tokens:
            return text, False

        truncated_ids = encoded["input_ids"][:self.config.max_tokens]
        truncated_text = self.tokenizer.decode(
            truncated_ids, skip_special_tokens=True)
        return truncated_text, True

    # -------------------------------------------------------------------------
    # Vector Store
    # -------------------------------------------------------------------------

    def embed_and_store(self, incremental: bool = True) -> FAISS:
        """Embed chunks and store in FAISS vector database."""
        if not self.chunks:
            raise ValueError("No chunks loaded. Call load_from_cache() first.")

        print(f"\n{'='*60}")
        print("EMBEDDING AND STORING")
        print(f"{'='*60}")
        print(f"Mode: {'Incremental' if incremental else 'Full rebuild'}")
        print(f"Chunks: {len(self.chunks)}")

        save_path = self.config.vector_db_path

        # Prepare texts
        texts, metadatas = [], []
        truncation_count = 0

        for chunk in tqdm(self.chunks, desc="Preparing"):
            text, was_truncated = self.truncate_to_max_tokens(
                chunk.page_content)
            if was_truncated:
                truncation_count += 1
            texts.append(text)
            metadatas.append(chunk.metadata)

        if truncation_count > 0:
            print(f"⚠️ Truncated {truncation_count} chunks")

        # Try loading existing vectorstore
        if incremental:
            try:
                existing_vs = FAISS.load_local(
                    folder_path=save_path,
                    embeddings=self.embedding_model,
                    allow_dangerous_deserialization=True,
                    distance_strategy=DistanceStrategy.COSINE,
                )
                print(
                    f"✓ Loaded existing vectorstore. Adding {len(texts)} chunks...")
                existing_vs.add_texts(texts=texts, metadatas=metadatas)
                self.vectorstore = existing_vs
            except Exception as e:
                print(f"No existing vectorstore ({e}). Creating new...")
                incremental = False

        if not incremental:
            print("Embedding chunks...")
            embeddings_list = []
            for text in tqdm(texts, desc="Embedding"):
                vec = self.embedding_model.embed_documents([text])[0]
                embeddings_list.append(vec)

            text_embedding_pairs = list(zip(texts, embeddings_list))
            self.vectorstore = FAISS.from_embeddings(
                text_embeddings=text_embedding_pairs,
                embedding=self.embedding_model,
                metadatas=metadatas,
                distance_strategy=DistanceStrategy.COSINE,
            )

        os.makedirs(self.config.vector_db_base, exist_ok=True)
        self.vectorstore.save_local(save_path)
        print(f"✓ Saved vectorstore to {save_path}")

        return self.vectorstore

    def load_vectorstore(self) -> FAISS:
        """Load existing vector store from disk."""
        print(f"Loading vectorstore from {self.config.vector_db_path}...")

        self.vectorstore = FAISS.load_local(
            folder_path=self.config.vector_db_path,
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True,
            distance_strategy=DistanceStrategy.COSINE,
        )

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config.retrieval_k}
        )

        print("✓ Vectorstore loaded")
        return self.vectorstore

    def setup_retriever(self):
        """Set up the retriever."""
        if self.vectorstore is None:
            raise ValueError(
                "No vectorstore. Call load_vectorstore() or embed_and_store() first.")

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config.retrieval_k}
        )
        print("✓ Retriever ready")

    # -------------------------------------------------------------------------
    # Full Pipeline
    # -------------------------------------------------------------------------

    def setup(self, incremental: bool = True):
        """Run full setup: load cache, embed, setup retriever."""
        self.load_from_cache()
        self.embed_and_store(incremental=incremental)
        self.setup_retriever()
        print("\n✓ Pipeline ready!")

    # -------------------------------------------------------------------------
    # Extraction
    # -------------------------------------------------------------------------

    def get_companies(self) -> List[str]:
        """Get sorted list of unique company IDs."""
        return sorted(set(
            c.metadata.get("company_id")
            for c in self.chunks
            if c.metadata.get("company_id")
        ))

    def get_years_for_company(self, company_id: str) -> List[str]:
        """Get sorted list of years for a specific company."""
        return sorted(set(
            c.metadata.get("year")
            for c in self.chunks
            if c.metadata.get("company_id") == company_id and c.metadata.get("year")
        ))

    def _parse_llm_response(self, response_text: str) -> List[str]:
        """Parse LLM response into list of items."""
        if not response_text:
            return []

        text = response_text.strip()
        if any(x in text.upper() for x in ["NONE_FOUND", "NO_BARRIERS", "NO_MOTIVATORS"]):
            return []

        items = []
        seen: Set[str] = set()

        for line in text.splitlines():
            line = line.strip()
            if not line or len(line) < 10:
                continue

            if line.startswith(("-", "•", "*", "–")):
                line = line[1:].strip()

            normalized = line.lower().strip()
            if normalized in seen:
                continue

            # Check word overlap for deduplication
            is_duplicate = False
            for existing in seen:
                existing_words = set(existing.split())
                line_words = set(normalized.split())
                overlap = len(existing_words & line_words)
                if overlap >= min(len(existing_words), len(line_words)) * 0.7:
                    is_duplicate = True
                    break

            if not is_duplicate:
                items.append(line)
                seen.add(normalized)

        return items

    def _extract_from_context(self, context: str, extract_type: str) -> List[str]:
        """Extract barriers or motivators from context using LLM."""
        prompt = BARRIER_PROMPT if extract_type == "barriers" else MOTIVATOR_PROMPT
        chain = prompt | self.llm

        try:
            response = chain.invoke({"context": context})
            raw_text = response.content if hasattr(
                response, "content") else str(response)
            return self._parse_llm_response(raw_text)
        except Exception as e:
            print(f"  ⚠️ Error extracting {extract_type}: {e}")
            return []

    def extract_company_data(self, company_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract barriers and motivators for a company across all years."""
        print(f"\n{'='*60}")
        print(f"Extracting: {company_id}")
        print(f"{'='*60}")

        years = self.get_years_for_company(company_id)
        print(f"Years: {years}")

        barrier_rows, motivator_rows = [], []

        for year in years:
            print(f"\nYear {year}...")

            # Get relevant chunks via semantic search
            if self.retriever:
                barrier_docs = [
                    d for d in self.retriever.invoke(BARRIER_QUERY)
                    if d.metadata.get("company_id") == company_id
                    and d.metadata.get("year") == year
                ][:self.config.context_chunks]

                motivator_docs = [
                    d for d in self.retriever.invoke(MOTIVATOR_QUERY)
                    if d.metadata.get("company_id") == company_id
                    and d.metadata.get("year") == year
                ][:self.config.context_chunks]
            else:
                # Fallback: use all chunks for this year
                year_chunks = [
                    c for c in self.chunks
                    if c.metadata.get("company_id") == company_id
                    and c.metadata.get("year") == year
                ]
                barrier_docs = year_chunks[:self.config.context_chunks * 2]
                motivator_docs = year_chunks[:self.config.context_chunks * 2]

            # Extract barriers
            barriers = []
            if barrier_docs:
                context = "\n\n---\n\n".join(
                    d.page_content for d in barrier_docs)
                barriers = self._extract_from_context(context, "barriers")
            print(f"  → {len(barriers)} barriers")

            # Extract motivators
            motivators = []
            if motivator_docs:
                context = "\n\n---\n\n".join(
                    d.page_content for d in motivator_docs)
                motivators = self._extract_from_context(context, "motivators")
            print(f"  → {len(motivators)} motivators")

            barrier_rows.append({
                "company_id": company_id,
                "year": year,
                "barriers": "\n".join(barriers) if barriers else "NONE_FOUND",
                "barrier_count": len(barriers),
            })
            motivator_rows.append({
                "company_id": company_id,
                "year": year,
                "motivators": "\n".join(motivators) if motivators else "NONE_FOUND",
                "motivator_count": len(motivators),
            })

        return pd.DataFrame(barrier_rows), pd.DataFrame(motivator_rows)

    def extract_all_companies(self, save_results: bool = True) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Extract barriers and motivators for all companies."""
        print(f"\n{'='*70}")
        print("EXTRACTING ALL COMPANIES")
        print(f"{'='*70}")

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

                print(f"✓ {company_id}")
            except Exception as e:
                print(f"✗ {company_id}: {e}")

        print(f"\n{'='*70}")
        print("✓ COMPLETE")
        print(f"{'='*70}")

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

    def inspect_status(self):
        """Show pipeline status."""
        print(f"\n{'='*60}")
        print("PIPELINE STATUS")
        print(f"{'='*60}")

        status = {
            "Chunks loaded": len(self.chunks) if self.chunks else 0,
            "Vectorstore": "✓" if self.vectorstore else "✗",
            "Retriever": "✓" if self.retriever else "✗",
            "LLM": "✓" if self._llm else "○ (lazy)",
        }

        for k, v in status.items():
            print(f"  {k}: {v}")

        print(f"\nConfig:")
        print(f"  Cache: {self.config.cache_dir}")
        print(f"  VectorDB: {self.config.vector_db_path}")
        print(f"  Ollama: {self.config.ollama_model}")

    def display_results(self, company_id: str, df_barriers: pd.DataFrame, df_motivators: pd.DataFrame):
        """Display extraction results."""
        print(f"\n{'='*60}")
        print(f"BARRIERS - {company_id}")
        print(f"{'='*60}")
        print(tabulate(df_barriers, headers='keys', tablefmt='grid'))

        print(f"\n{'='*60}")
        print(f"MOTIVATORS - {company_id}")
        print(f"{'='*60}")
        print(tabulate(df_motivators, headers='keys', tablefmt='grid'))

    def cleanup(self):
        """Release resources."""
        self.gpu.clear()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_start(
    cache_dir: str = "cache",
    use_bert_cache: bool = False,
    incremental: bool = True,
    ollama_model: str = "llama3.1:8b"
) -> RAGPipeline:
    """Quick start: load cache, embed, setup retriever."""
    config = RAGConfig(
        cache_dir=cache_dir,
        use_bert_cache=use_bert_cache,
        ollama_model=ollama_model,
    )
    pipeline = RAGPipeline(config)
    pipeline.setup(incremental=incremental)
    return pipeline


def load_existing(
    vector_db_path: str = "vector_db/reports",
    cache_dir: str = "cache",
    ollama_model: str = "llama3.1:8b"
) -> RAGPipeline:
    """Load existing vectorstore and cache."""
    config = RAGConfig(
        cache_dir=cache_dir,
        ollama_model=ollama_model,
    )
    config.vector_db_base = os.path.dirname(vector_db_path)
    config.vector_db_name = os.path.basename(vector_db_path)

    pipeline = RAGPipeline(config)
    pipeline.load_from_cache()
    pipeline.load_vectorstore()

    return pipeline


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # print("RAG Pipeline for Decarbonisation Report Analysis")
    # print("=" * 60)
    # print("\nUsage:")
    # print("  from rag_pipeline import RAGPipeline, quick_start")
    # print()
    # print("  # Quick start (loads from prep.json cache)")
    # print("  pipeline = quick_start('cache')")
    # print("  pipeline.extract_all_companies()")
    # print()
    # print("  # Use BERT cache (with classification scores)")
    # print("  pipeline = quick_start('cache', use_bert_cache=True)")
    # print()
    # print("  # Manual setup")
    # print("  pipeline = RAGPipeline()")
    # print("  pipeline.load_from_cache()")
    # print("  pipeline.embed_and_store()")
    # print("  pipeline.setup_retriever()")
    # print("  pipeline.inspect_status()")
    # print()
    # print("NOTE: Ensure Ollama is running:")
    # print("  ollama run llama3.1:8b")

    # Manual setup
    pipeline = RAGPipeline()
    # pipeline.load_from_cache()
    # pipeline.embed_and_store()
    pipeline.setup_retriever()
    pipeline.inspect_status()
