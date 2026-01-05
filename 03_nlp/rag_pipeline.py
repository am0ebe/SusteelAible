"""
RAG Pipeline for Company Decarbonisation Report Analysis
========================================================

This module provides a complete pipeline for:
1. Loading and preprocessing PDFs (via shared preprocessing module)
2. Creating/updating FAISS vector stores
3. Extracting barriers and motivators using local LLM (Ollama)

Key improvements:
- Uses Ollama for local inference with GPU support
- Improved prompts to reduce hallucinations
- Automatic deduplication of extracted items
- Simplified extraction logic

Usage:
    from rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()
    pipeline.load_and_process_pdfs()
    pipeline.extract_all_companies()
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from tabulate import tabulate

# from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_ollama import ChatOllama
from transformers import AutoTokenizer

from preprocessing import (
    PDFPreprocessor,
    PreprocessingConfig,
    ProcessedDocument,
    get_gpu_info,
)

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline."""

    # Paths
    reports_folder: str = "../data/reports"
    vector_db_base: str = "vector_db"
    vector_db_name: str = "reports"
    output_folder: str = "decarbonisation_tables"

    # Embedding model
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    max_tokens: int = 512

    # LLM settings (Ollama)
    ollama_model: str = "llama3.1:8b"  # or "mistral", "llama3.2", etc.
    ollama_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.0
    llm_num_ctx: int = 4096  # Context window - shortterm mem tokens. # try 8092 if GPU good

    # Retrieval
    retrieval_k: int = 50 # most simi, cheap
    context_chunks: int = 5

    # Preprocessing
    chunk_method: str = "semantic"
    min_chunk_chars: int = 600
    max_chunk_chars: int = 1600
    translate_to_english: bool = True

    @property
    def vector_db_path(self) -> str:
        return os.path.join(self.vector_db_base, self.vector_db_name)

    def get_preprocessing_config(self) -> PreprocessingConfig:
        """Create PreprocessingConfig from RAG settings."""
        return PreprocessingConfig(
            min_chunk_chars=self.min_chunk_chars,
            max_chunk_chars=self.max_chunk_chars,
        )


# =============================================================================
# PROMPTS - Designed to reduce hallucinations and over-generalization
# =============================================================================

BARRIER_PROMPT = ChatPromptTemplate.from_template("""You are analyzing company sustainability reports to extract BARRIERS to decarbonisation.

TASK: Extract specific barriers mentioned in the text below. Only extract barriers that are EXPLICITLY stated or clearly implied.

RULES:
1. Each barrier must be a specific, concrete challenge (not vague statements)
2. Only include barriers actually mentioned in the context - DO NOT invent or assume barriers
3. If a barrier is mentioned multiple times, include it only once
4. Use concise phrasing (5-15 words per barrier)
5. If no clear barriers are found, respond with exactly: NONE_FOUND

EXAMPLES OF GOOD BARRIERS:
- High upfront costs for renewable energy infrastructure
- Lack of skilled workforce for green technology implementation
- Regulatory uncertainty around carbon pricing
- Supply chain dependency on carbon-intensive materials

EXAMPLES OF BAD BARRIERS (too vague/generic):
- Climate change is challenging
- There are many obstacles
- Sustainability is difficult

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

EXAMPLES OF GOOD MOTIVATORS:
- Cost savings from energy efficiency improvements
- Regulatory requirements for emissions reporting
- Customer demand for sustainable products
- Commitment to net-zero by 2050

EXAMPLES OF BAD MOTIVATORS (too vague/generic):
- Sustainability is important
- Want to be green
- Environmental concerns

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
        """Initialize the pipeline with optional custom configuration."""
        self.config = config or RAGConfig()

        self.preprocessor = PDFPreprocessor(
            self.config.get_preprocessing_config())

        # Lazy-loaded components
        self._llm = None
        self._embedding_model = None
        self._tokenizer = None

        # Data containers
        self.documents: List[ProcessedDocument] = []
        self.chunks: List[Document] = []
        self.vectorstore: Optional[FAISS] = None
        self.retriever = None

        # Print GPU info
        gpu_info = get_gpu_info()
        if gpu_info:
            print(
                f"✓ GPU available: {gpu_info['name']} ({gpu_info['total_memory_gb']:.1f}GB)")
        else:
            print("✓ Running on CPU")

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
    # PDF Loading
    # -------------------------------------------------------------------------

    def load_pdfs(self, folder: Optional[str] = None) -> List[ProcessedDocument]:
        """Load and preprocess all PDFs from the reports folder."""
        folder = folder or self.config.reports_folder

        self.documents = self.preprocessor.process_folder(
            folder,
            chunk_method=self.config.chunk_method,
            translate_to_english=self.config.translate_to_english,
            recursive=True,
            show_progress=True,
        )

        self.chunks = self.preprocessor.to_langchain_documents(self.documents)
        print(f"✓ Created {len(self.chunks)} LangChain chunks")

        return self.documents

    # -------------------------------------------------------------------------
    # Token Management
    # -------------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        encoded = self.tokenizer(
            text, add_special_tokens=False, truncation=False)
        return len(encoded["input_ids"])

    def truncate_to_max_tokens(self, text: str) -> Tuple[str, bool]:
        """Truncate text to max tokens if needed."""
        n_tokens = self.count_tokens(text)

        if n_tokens <= self.config.max_tokens:
            return text, False

        encoded = self.tokenizer(
            text, add_special_tokens=False, truncation=False)
        truncated_ids = encoded["input_ids"][:self.config.max_tokens]
        truncated_text = self.tokenizer.decode(
            truncated_ids, skip_special_tokens=True)
        return truncated_text, True

    # -------------------------------------------------------------------------
    # Vector Store Management
    # -------------------------------------------------------------------------

    def embed_and_store(
        self,
        chunks: Optional[List[Document]] = None,
        incremental: bool = True
    ) -> FAISS:
        """Embed chunks and store in FAISS vector database."""
        chunks = chunks or self.chunks

        if not chunks:
            raise ValueError("No chunks to embed. Call load_pdfs() first.")

        print(f"\n{'='*60}")
        print("EMBEDDING AND STORING")
        print(f"{'='*60}")
        print(f"Mode: {'Incremental' if incremental else 'Full rebuild'}")
        print(f"Chunks to process: {len(chunks)}")

        save_path = self.config.vector_db_path

        # Prepare texts
        texts = []
        metadatas = []
        truncation_count = 0

        for chunk in tqdm(chunks, desc="Preparing"):
            text, was_truncated = self.truncate_to_max_tokens(
                chunk.page_content)
            if was_truncated:
                truncation_count += 1
            texts.append(text)
            metadatas.append(chunk.metadata)

        if truncation_count > 0:
            print(
                f"⚠️ Truncated {truncation_count} chunks to {self.config.max_tokens} tokens")

        # Try to load existing vectorstore
        if incremental:
            try:
                existing_vs = FAISS.load_local(
                    folder_path=save_path,
                    embeddings=self.embedding_model,
                    allow_dangerous_deserialization=True, # << 2D whats this?
                    distance_strategy=DistanceStrategy.COSINE,
                )
                print(
                    f"✓ Loaded existing vectorstore. Adding {len(texts)} chunks...")
                existing_vs.add_texts(texts=texts, metadatas=metadatas)
                self.vectorstore = existing_vs
            except Exception as e:
                print(f"No existing vectorstore found ({e}). Creating new...")
                incremental = False

        if not incremental:
            print("\nEmbedding chunks...")
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
        print(f"\n✓ Saved vectorstore to {save_path}")

        return self.vectorstore

    def load_vectorstore(self) -> FAISS:
        """Load existing vector store from disk."""
        print(f"\nLoading vectorstore from {self.config.vector_db_path}...")

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

    def load_and_process_pdfs(self, incremental: bool = True):
        """Run the full loading and processing pipeline."""
        self.load_pdfs()
        self.embed_and_store(incremental=incremental)
        self.setup_retriever()
        print("\n✓ Pipeline ready!")

    # -------------------------------------------------------------------------
    # Extraction Functions
    # -------------------------------------------------------------------------

    def get_companies(self) -> List[str]:
        """Get sorted list of unique company IDs."""
        if self.chunks:
            return sorted(set(
                c.metadata.get("company_id")
                for c in self.chunks
                if c.metadata.get("company_id")
            ))
        return []

    def get_years_for_company(self, company_id: str) -> List[str]:
        """Get sorted list of years for a specific company."""
        return sorted(set(
            c.metadata.get("year")
            for c in self.chunks
            if c.metadata.get("company_id") == company_id
            and c.metadata.get("year")
        ))

    def _parse_llm_response(self, response_text: str) -> List[str]:
        """Parse LLM response into list of items, removing duplicates."""
        if not response_text:
            return []

        text = response_text.strip()

        # Check for "none found" indicators
        if any(x in text.upper() for x in ["NONE_FOUND", "NO_BARRIERS", "NO_MOTIVATORS", "NONE FOUND"]):
            return []

        items = []
        seen: Set[str] = set()

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            # Remove bullet point markers
            if line.startswith(("-", "•", "*", "–")):
                line = line[1:].strip()

            # Skip very short or generic items
            if len(line) < 10:
                continue

            # Normalize for deduplication
            normalized = line.lower().strip()

            # Skip if too similar to existing item
            if normalized in seen:
                continue

            # Check for substantial overlap with existing items
            is_duplicate = False
            for existing in seen:
                # Simple word overlap check
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

    def _extract_from_context(
        self,
        context: str,
        extract_type: str
    ) -> List[str]:
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

    def extract_company_data(
        self,
        company_id: str,
        use_semantic_search: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract barriers and motivators for a company across all years."""
        print(f"\n{'='*60}")
        print(f"Extracting data for Company {company_id}")
        print(f"{'='*60}")

        years = self.get_years_for_company(company_id)
        print(f"Years found: {years}")

        barrier_rows = []
        motivator_rows = []

        for year in years:
            print(f"\nProcessing year {year}...")

            # Get relevant chunks
            if use_semantic_search and self.retriever:
                # Semantic search for barriers
                barrier_docs = self.retriever.invoke(BARRIER_QUERY)
                barrier_docs = [
                    d for d in barrier_docs
                    if d.metadata.get("company_id") == company_id
                    and d.metadata.get("year") == year
                ][:self.config.context_chunks]

                # Semantic search for motivators
                motivator_docs = self.retriever.invoke(MOTIVATOR_QUERY)
                motivator_docs = [
                    d for d in motivator_docs
                    if d.metadata.get("company_id") == company_id
                    and d.metadata.get("year") == year
                ][:self.config.context_chunks]
            else:
                # Use all chunks for this year
                year_chunks = [
                    c for c in self.chunks
                    if c.metadata.get("company_id") == company_id
                    and c.metadata.get("year") == year
                ]
                barrier_docs = year_chunks[:self.config.context_chunks * 2]
                motivator_docs = year_chunks[:self.config.context_chunks * 2]

            # Extract barriers
            if barrier_docs:
                context = "\n\n---\n\n".join(
                    d.page_content for d in barrier_docs)
                barriers = self._extract_from_context(context, "barriers")
                print(f"  → Found {len(barriers)} barriers")
            else:
                barriers = []
                print("  → No relevant barrier documents found")

            # Extract motivators
            if motivator_docs:
                context = "\n\n---\n\n".join(
                    d.page_content for d in motivator_docs)
                motivators = self._extract_from_context(context, "motivators")
                print(f"  → Found {len(motivators)} motivators")
            else:
                motivators = []
                print("  → No relevant motivator documents found")

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

    def extract_all_companies(
        self,
        use_semantic_search: bool = True,
        save_results: bool = True
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Extract barriers and motivators for all companies."""
        print(f"\n{'='*70}")
        print("EXTRACTING DATA FOR ALL COMPANIES")
        print(f"{'='*70}")

        results = {}
        companies = self.get_companies()

        for company_id in companies:
            try:
                df_barriers, df_motivators = self.extract_company_data(
                    company_id,
                    use_semantic_search=use_semantic_search
                )
                results[company_id] = (df_barriers, df_motivators)

                if save_results:
                    self.save_company_tables(
                        company_id, df_barriers, df_motivators)

                print(f"✓ Completed {company_id}\n")
            except Exception as e:
                print(f"✗ Error with {company_id}: {e}\n")

        print(f"\n{'='*70}")
        print("✓ ALL COMPANIES PROCESSED!")
        print(f"{'='*70}")

        return results

    def save_company_tables(
        self,
        company_id: str,
        df_barriers: pd.DataFrame,
        df_motivators: pd.DataFrame
    ):
        """Save extraction results to CSV and Excel."""
        os.makedirs(self.config.output_folder, exist_ok=True)

        base_barrier = os.path.join(
            self.config.output_folder,
            f"barriers_company_{company_id}"
        )
        base_motivator = os.path.join(
            self.config.output_folder,
            f"motivators_company_{company_id}"
        )

        df_barriers.to_csv(f"{base_barrier}.csv", index=False)
        df_barriers.to_excel(f"{base_barrier}.xlsx", index=False)

        df_motivators.to_csv(f"{base_motivator}.csv", index=False)
        df_motivators.to_excel(f"{base_motivator}.xlsx", index=False)

        print(f"\n✓ Saved to {self.config.output_folder}/")

    # -------------------------------------------------------------------------
    # INSPECTION FUNCTIONS
    # -------------------------------------------------------------------------

    def inspect_documents(self, sample_size: int = 3):
        """Inspect loaded documents."""
        print(f"\n{'='*60}")
        print("DOCUMENT INSPECTION")
        print(f"{'='*60}")

        if not self.documents:
            print("⚠️ No documents loaded. Call load_pdfs() first.")
            return

        print(f"\nTotal documents: {len(self.documents)}")

        company_counts = {}
        for doc in self.documents:
            cid = doc.company_id or "Unknown"
            company_counts[cid] = company_counts.get(cid, 0) + 1

        print(f"\nDocuments per company:")
        for cid, count in sorted(company_counts.items()):
            print(f"  {cid}: {count}")

        lang_counts = {}
        for doc in self.documents:
            lang = doc.language
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        print(f"\nLanguage distribution:")
        for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
            print(f"  {lang}: {count}")

        translated_count = sum(1 for d in self.documents if d.translated)
        print(f"\nTranslated documents: {translated_count}")

    def inspect_chunks(self, sample_size: int = 3):
        """Inspect chunked documents."""
        print(f"\n{'='*60}")
        print("CHUNK INSPECTION")
        print(f"{'='*60}")

        if not self.chunks:
            print("⚠️ No chunks created. Call load_pdfs() first.")
            return

        print(f"\nTotal chunks: {len(self.chunks)}")

        company_year_counts = {}
        for chunk in self.chunks:
            key = (
                chunk.metadata.get("company_id", "Unknown"),
                chunk.metadata.get("year", "Unknown")
            )
            company_year_counts[key] = company_year_counts.get(key, 0) + 1

        print(f"\nChunks per company/year:")
        for (cid, year), count in sorted(company_year_counts.items()):
            print(f"  {cid} / {year}: {count} chunks")

    def inspect_pipeline_status(self):
        """Show overall pipeline status."""
        print(f"\n{'='*60}")
        print("PIPELINE STATUS")
        print(f"{'='*60}")

        status = {
            "Documents loaded": len(self.documents) if self.documents else 0,
            "Chunks created": len(self.chunks) if self.chunks else 0,
            "Vectorstore": "✓ Ready" if self.vectorstore else "✗ Not loaded",
            "Retriever": "✓ Ready" if self.retriever else "✗ Not configured",
            "LLM": "✓ Ready" if self._llm else "○ Not initialized (lazy)",
            "Embedding model": "✓ Ready" if self._embedding_model else "○ Not initialized (lazy)",
        }

        print("\nComponent Status:")
        for component, state in status.items():
            print(f"  {component}: {state}")

        print(f"\nConfiguration:")
        print(f"  Reports folder: {self.config.reports_folder}")
        print(f"  Vector DB path: {self.config.vector_db_path}")
        print(f"  Ollama model: {self.config.ollama_model}")
        print(
            f"  Translation: {'enabled' if self.config.translate_to_english else 'disabled'}")

    def display_results(
        self,
        company_id: str,
        df_barriers: pd.DataFrame,
        df_motivators: pd.DataFrame
    ):
        """Display extraction results in table format."""
        print(f"\n{'='*60}")
        print(f"BARRIERS - Company {company_id}")
        print(f"{'='*60}")
        print(tabulate(df_barriers, headers='keys', tablefmt='grid'))

        print(f"\n{'='*60}")
        print(f"MOTIVATORS - Company {company_id}")
        print(f"{'='*60}")
        print(tabulate(df_motivators, headers='keys', tablefmt='grid'))

    def cleanup(self):
        """Release all resources."""
        self.preprocessor.cleanup()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_start(
    reports_folder: str = "../data/reports",
    incremental: bool = True,
    ollama_model: str = "llama3.1:8b"
) -> RAGPipeline:
    """Quick start function to get the pipeline running."""
    config = RAGConfig(
        reports_folder=reports_folder,
        ollama_model=ollama_model,
    )
    pipeline = RAGPipeline(config)
    pipeline.load_and_process_pdfs(incremental=incremental)
    return pipeline


def load_existing_pipeline(
    vector_db_path: str = "vector_db/reports",
    ollama_model: str = "llama3.1:8b"
) -> RAGPipeline:
    """Load an existing pipeline from a saved vector store."""
    config = RAGConfig(ollama_model=ollama_model)
    config.vector_db_base = os.path.dirname(vector_db_path)
    config.vector_db_name = os.path.basename(vector_db_path)

    pipeline = RAGPipeline(config)
    pipeline.load_vectorstore()

    return pipeline


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("RAG Pipeline for Decarbonisation Report Analysis")
    print("=" * 60)
    print("\nUsage:")
    print("  from rag_pipeline import RAGPipeline, quick_start")
    print()
    print("  # Full pipeline")
    print("  pipeline = RAGPipeline()")
    print("  pipeline.load_and_process_pdfs()")
    print("  pipeline.inspect_pipeline_status()")
    print("  pipeline.extract_all_companies()")
    print()
    print("  # Quick start")
    print("  pipeline = quick_start('../data/reports')")
    print()
    print("NOTE: Ensure Ollama is running with your preferred model:")
    print("  ollama run llama3.1:8b")
