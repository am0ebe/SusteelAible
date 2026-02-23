"""
RAG Pipeline with Embedding Retrieval for Decarbonisation Report Analysis
=========================================================================

Real RAG: embeds all chunks → FAISS index → retrieves top_k per query → LLM extract.
Extends ExtractPipeline (llm_extract.py) with targeted retrieval.

Usage:
    from nlp import Config, load_pipeline

    config = Config(
        approach="rag", llm_provider="groq", model="llama-3.1-8b-instant",
        ctx=128000, batch_size=3, top_k=20,
    )
    pipeline = load_pipeline(config)
    pipeline.extract_all_companies()
"""

import os
import time
from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from nlp.llm_extract import Config, ExtractPipeline


os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
# RETRIEVAL QUERIES
# =============================================================================

# 3 queries per category: each angles at a different facet of the concept.
# FAISS retrieves top_k chunks per query; the union is deduplicated before the LLM call.
# Diversity of query angles improves recall — a single query misses chunks that
# use different vocabulary for the same underlying concept.
BARRIER_QUERIES = [
    "barriers challenges constraints to decarbonisation reducing GHG emissions",
    "costs risks difficulties limiting net-zero carbon reduction targets",
    "obstacles preventing fossil fuel transition green steel hydrogen",
]

MOTIVATOR_QUERIES = [
    "motivators drivers incentives for decarbonisation reducing emissions",
    "commitments targets policies supporting net-zero transition",
    "opportunities benefits pressures encouraging green steel low-carbon",
]


# =============================================================================
# MAIN PIPELINE CLASS
# =============================================================================

class RAGPipeline(ExtractPipeline):
    """RAG pipeline with FAISS embedding retrieval."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.vectorstore: Optional[FAISS] = None
        self._embeddings = None

    def _init_embeddings(self):
        """Lazy load embedding model."""
        if self._embeddings is None:
            print(f"Loading embedding model: {self.config.embedding_model}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={"device": str(self.gpu.device)},
            )
        return self._embeddings

    def _build_or_load_faiss(self):
        """Build FAISS index from chunks, or load from cache."""
        cache_path = self.config.faiss_cache_path

        # Try loading cached index
        if cache_path and os.path.exists(cache_path) and self.config.reuse_faiss_cache:
            print(f"Loading FAISS index from {cache_path}")
            embeddings = self._init_embeddings()
            self.vectorstore = FAISS.load_local(
                cache_path, embeddings, allow_dangerous_deserialization=True
            )
            print(
                f"✓ FAISS index loaded ({self.vectorstore.index.ntotal} vectors)")
            return

        # Build new index
        embeddings = self._init_embeddings()
        print(f"Building FAISS index from {len(self.chunks)} chunks...")
        start = time.time()

        self.vectorstore = FAISS.from_documents(self.chunks, embeddings)

        elapsed = time.time() - start
        print(
            f"✓ FAISS index built ({self.vectorstore.index.ntotal} vectors, {elapsed:.1f}s)")

        # Save to cache
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            self.vectorstore.save_local(cache_path)
            print(f"✓ FAISS index saved to {cache_path}")

    def load_from_cache(self) -> List:
        """Load documents from cache, then build/load FAISS index."""
        chunks = super().load_from_cache()
        self._build_or_load_faiss()
        return chunks

    def _retrieve_chunks(self, company_id: str, year: str, extract_type: str) -> List:
        """Retrieve relevant chunks for a company-year using embedding similarity."""
        queries = BARRIER_QUERIES if extract_type == "barriers" else MOTIVATOR_QUERIES
        filter_dict = {"company_id": company_id, "year": str(year)}
        per_query_k = max(self.config.top_k // len(queries), 1)

        all_chunks, seen_ids = [], set()
        for query in queries:
            if self.config.retrieval_strategy == "mmr":
                results = self.vectorstore.max_marginal_relevance_search(
                    query, k=per_query_k, fetch_k=self.config.mmr_fetch_k,
                    lambda_mult=self.config.mmr_lambda, filter=filter_dict,
                )
            else:
                results = self.vectorstore.similarity_search(
                    query, k=per_query_k, filter=filter_dict,
                )

            for doc in results:
                cid = doc.metadata.get("chunk_id")
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    all_chunks.append(doc)

        return all_chunks[:self.config.top_k]

    def extract_company_year(self, company_id: str, year: str) -> Tuple[List[str], List[str]]:
        """Extract barriers and motivators using retrieval instead of exhaustive batching."""
        company_name = self._get_company_name(company_id)

        barrier_chunks = self._retrieve_chunks(company_id, year, "barriers")
        motivator_chunks = self._retrieve_chunks(
            company_id, year, "motivators")

        if not barrier_chunks and not motivator_chunks:
            return [], []

        barriers = self._map_extract(
            barrier_chunks, company_name, year, "barriers")
        motivators = self._map_extract(
            motivator_chunks, company_name, year, "motivators")

        return barriers, motivators

    def extract_company_data(self, company_id: str) -> Tuple:
        """Extract barriers and motivators for a company across all years."""
        company_name = self._get_company_name(company_id)

        print(f"\n{'='*60}")
        print(f"Extracting: {company_name} ({company_id})")
        print(f"{'='*60}")

        years = self.get_years_for_company(company_id)
        print(f"Years: {years}")
        print(
            f"Retrieval: top_k={self.config.top_k}, strategy={self.config.retrieval_strategy}")

        barrier_rows, motivator_rows = [], []

        for year in tqdm(years, desc=f"  {company_id}", leave=False):
            barriers, motivators = self.extract_company_year(company_id, year)

            for text in barriers:
                barrier_rows.append({
                    "company_id": company_id,
                    "company": company_name,
                    "year": year,
                    "barriers": text,
                })

            for text in motivators:
                motivator_rows.append({
                    "company_id": company_id,
                    "company": company_name,
                    "year": year,
                    "motivators": text,
                })

        print(
            f"  ✓ Extracted {len(barrier_rows)} barriers, {len(motivator_rows)} motivators")

        return pd.DataFrame(barrier_rows), pd.DataFrame(motivator_rows)

    def cleanup(self):
        """Release resources including embedding model."""
        self.vectorstore = None
        self._embeddings = None
        super().cleanup()
