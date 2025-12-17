"""
RAG Pipeline for Company Decarbonisation Report Analysis
========================================================

This module provides a complete pipeline for:
1. Loading PDFs from company report folders
2. Chunking documents with metadata preservation
3. Creating/updating FAISS vector stores (incremental)
4. Extracting barriers and motivators using LLM chains
5. Inspection utilities to verify pipeline stages

Usage:
    from rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()
    pipeline.load_and_process_pdfs()
    pipeline.inspect_chunks()
    pipeline.extract_all_companies()
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

import pandas as pd
import fitz  # PyMuPDF
from dotenv import load_dotenv
from tqdm import tqdm
from tabulate import tabulate

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from transformers import AutoTokenizer

# Load environment variables
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PipelineConfig:
    """Central configuration for the RAG pipeline."""

    # Paths
    reports_folder: str = "../data/reports"
    vector_db_base: str = "vector_db"
    vector_db_name: str = "reports"
    output_folder: str = "decarbonisation_tables"

    # Embedding model
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    max_tokens: int = 512

    # LLM settings
    llm_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0
    llm_max_retries: int = 2

    # Chunking
    chunk_size: int = 1500
    chunk_overlap: int = 100

    # Retrieval
    retrieval_k: int = 50  # Number of docs to retrieve before filtering
    context_chunks: int = 5  # Number of chunks to use for LLM context

    @property
    def vector_db_path(self) -> str:
        return os.path.join(self.vector_db_base, self.vector_db_name)


# =============================================================================
# PROMPTS
# =============================================================================

BARRIER_PROMPT = ChatPromptTemplate.from_template("""
You are extracting *barriers to decarbonisation* mentioned in company reports.
Given the context below, output ONLY a list of barriers as short bullet points.

RULES:
- Each barrier MUST be on its own line.
- Each line MUST start with "- ".
- Do NOT write any introductions, explanations, summaries, or extra text.
- If there are no clear barriers, output exactly: "NO_BARRIERS_FOUND".

Context:
{context}

Answer:
""")

MOTIVATOR_PROMPT = ChatPromptTemplate.from_template("""
You are extracting *motivators to decarbonise* mentioned in company reports.
Given the context below, output ONLY a list of motivators as short bullet points.

RULES:
- Each motivator MUST be on its own line.
- Each line MUST start with "- ".
- Do NOT write any introductions, explanations, summaries, or extra text.
- If there are no clear motivators, output exactly: "NO_MOTIVATORS_FOUND".

Context:
{context}

Answer:
""")

# Semantic search queries for relevant chunk retrieval
BARRIER_QUERY = (
    "barriers challenges obstacles difficulties problems constraints "
    "to decarbonization carbon reduction emissions reduction climate action"
)
MOTIVATOR_QUERY = (
    "motivators drivers incentives benefits opportunities reasons goals "
    "targets commitments for decarbonization carbon reduction emissions "
    "reduction climate action sustainability"
)


# =============================================================================
# MAIN PIPELINE CLASS
# =============================================================================

class RAGPipeline:
    """
    Complete RAG pipeline for company decarbonisation report analysis.

    Attributes:
        config: Pipeline configuration
        llm: Language model for extraction
        embedding_model: Embedding model for vectorization
        tokenizer: Tokenizer for token counting
        chunks: List of document chunks
        vectorstore: FAISS vector store
        retriever: Vector store retriever
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the pipeline with optional custom configuration."""
        self.config = config or PipelineConfig()

        # Initialize models (lazy loading pattern - only when needed)
        self._llm = None
        self._embedding_model = None
        self._tokenizer = None
        self._barrier_chain = None
        self._motivator_chain = None

        # Data containers
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.vectorstore: Optional[FAISS] = None
        self.retriever = None
        self.retrieval_chain = None

    # -------------------------------------------------------------------------
    # Lazy-loaded properties
    # -------------------------------------------------------------------------

    @property
    def llm(self):
        """Lazy-load the LLM."""
        if self._llm is None:
            self._llm = ChatGroq(
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                max_tokens=None,
                timeout=None,
                max_retries=self.config.llm_max_retries,
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

    @property
    def barrier_chain(self):
        """Lazy-load the barrier extraction chain."""
        if self._barrier_chain is None:
            self._barrier_chain = BARRIER_PROMPT | self.llm
        return self._barrier_chain

    @property
    def motivator_chain(self):
        """Lazy-load the motivator extraction chain."""
        if self._motivator_chain is None:
            self._motivator_chain = MOTIVATOR_PROMPT | self.llm
        return self._motivator_chain

    # -------------------------------------------------------------------------
    # PDF Loading
    # -------------------------------------------------------------------------

    def load_pdfs(self, folder: Optional[str] = None) -> List[Document]:
        """
        Load all PDFs from the reports folder structure.

        Expected structure:
            reports/
                CompanyA/
                    C001_2022_annual_report.pdf
                    C001_2023_annual_report.pdf
                CompanyB/
                    C002_2022_sustainability.pdf

        Returns:
            List of Document objects with metadata
        """
        folder = folder or self.config.reports_folder
        documents = []

        print(f"\n{'='*60}")
        print("LOADING PDFs")
        print(f"{'='*60}")
        print(f"Source folder: {folder}")

        for company_folder in os.listdir(folder):
            company_path = os.path.join(folder, company_folder)

            if not os.path.isdir(company_path):
                continue

            company_name = company_folder

            for filename in os.listdir(company_path):
                if not filename.lower().endswith(".pdf"):
                    continue

                full_path = os.path.join(company_path, filename)
                print(f"  Loading: {filename}")

                try:
                    doc = fitz.open(full_path)
                    text = ""
                    for page_num in range(len(doc)):
                        text += doc[page_num].get_text()
                    doc.close()
                except Exception as e:
                    print(f"  ⚠️ Skipping {filename}: {e}")
                    continue

                # Parse metadata from filename
                name, _ = os.path.splitext(filename)
                parts = name.split("_")
                company_id = parts[0] if len(parts) > 0 else None
                year = parts[1] if len(parts) > 1 else None
                report_type = "_".join(parts[2:]) if len(parts) > 2 else None

                doc_obj = Document(
                    page_content=text,
                    metadata={
                        "company_name": company_name,
                        "company_id": company_id,
                        "year": year,
                        "report_type": report_type,
                        "filename": filename,
                        "source_path": full_path,
                    }
                )
                documents.append(doc_obj)

        self.documents = documents
        print(f"\n✓ Loaded {len(documents)} documents")
        return documents

    # -------------------------------------------------------------------------
    # Chunking
    # -------------------------------------------------------------------------

    def chunk_documents(
        self,
        documents: Optional[List[Document]] = None
    ) -> List[Document]:
        """
        Split documents into chunks for embedding.

        Args:
            documents: Documents to chunk (uses self.documents if None)

        Returns:
            List of chunked Document objects
        """
        documents = documents or self.documents

        if not documents:
            raise ValueError("No documents to chunk. Call load_pdfs() first.")

        print(f"\n{'='*60}")
        print("CHUNKING DOCUMENTS")
        print(f"{'='*60}")
        print(f"Chunk size: {self.config.chunk_size}")
        print(f"Chunk overlap: {self.config.chunk_overlap}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        chunks = text_splitter.split_documents(documents)

        # Add unique chunk IDs
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"chunk_{i}"

        self.chunks = chunks
        print(f"\n✓ Created {len(chunks)} chunks")
        return chunks

    # -------------------------------------------------------------------------
    # Token Management
    # -------------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )
        return len(encoded["input_ids"])

    def truncate_to_max_tokens(self, text: str) -> Tuple[str, bool]:
        """
        Truncate text to max tokens if needed.

        Returns:
            Tuple of (text, was_truncated)
        """
        n_tokens = self.count_tokens(text)

        if n_tokens <= self.config.max_tokens:
            return text, False

        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )
        truncated_ids = encoded["input_ids"][:self.config.max_tokens]
        truncated_text = self.tokenizer.decode(
            truncated_ids,
            skip_special_tokens=True
        )
        return truncated_text, True

    # -------------------------------------------------------------------------
    # Vector Store Management
    # -------------------------------------------------------------------------

    def embed_and_store(
        self,
        chunks: Optional[List[Document]] = None,
        incremental: bool = True
    ) -> FAISS:
        """
        Embed chunks and store in FAISS vector database.

        Args:
            chunks: Chunks to embed (uses self.chunks if None)
            incremental: If True, add to existing DB; if False, overwrite

        Returns:
            FAISS vectorstore
        """
        chunks = chunks or self.chunks

        if not chunks:
            raise ValueError(
                "No chunks to embed. Call chunk_documents() first.")

        print(f"\n{'='*60}")
        print("EMBEDDING AND STORING")
        print(f"{'='*60}")
        print(f"Mode: {'Incremental' if incremental else 'Full rebuild'}")
        print(f"Chunks to process: {len(chunks)}")

        save_path = self.config.vector_db_path

        # Prepare texts with token truncation
        texts = []
        metadatas = []
        truncation_count = 0

        print("\nPreparing texts...")
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

        # Try to load existing vectorstore if incremental
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
                print(f"No existing vectorstore found ({e}). Creating new...")
                incremental = False

        if not incremental:
            # Create new vectorstore
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

        # Save to disk
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

    def setup_retrieval_chain(self):
        """Set up the retrieval chain for QA."""
        if self.retriever is None:
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.config.retrieval_k}
            )

        stuff_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=hub.pull("langchain-ai/retrieval-qa-chat")
        )

        self.retrieval_chain = create_retrieval_chain(
            retriever=self.retriever,
            combine_docs_chain=stuff_chain
        )

        print("✓ Retrieval chain ready")
        return self.retrieval_chain

    # -------------------------------------------------------------------------
    # Full Pipeline
    # -------------------------------------------------------------------------

    def load_and_process_pdfs(self, incremental: bool = True):
        """
        Run the full loading and processing pipeline.

        Args:
            incremental: If True, add to existing vector DB
        """
        self.load_pdfs()
        self.chunk_documents()
        self.embed_and_store(incremental=incremental)
        self.setup_retrieval_chain()
        print("\n✓ Pipeline ready!")

    # -------------------------------------------------------------------------
    # Extraction Functions
    # -------------------------------------------------------------------------

    @staticmethod
    def clean_bullet_list(text: str, empty_token: str = "NO_ITEMS_FOUND") -> str:
        """
        Clean LLM-generated bullet lists.

        - Returns empty_token if no usable bullets found
        - Accepts bullets starting with -, •, *
        - Normalizes to clean newline-separated items
        """
        if not text or not isinstance(text, str):
            return empty_token

        text = text.strip()

        if text.upper() == empty_token.upper():
            return empty_token

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned = []

        for line in lines:
            if line.startswith(("-", "•", "*")):
                item = line.lstrip("-•* ").strip()
                if item:
                    cleaned.append(item)

        return "\n".join(cleaned) if cleaned else empty_token

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

    def extract_company_data(
        self,
        company_id: str,
        use_semantic_search: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract barriers and motivators for a company across all years.

        Args:
            company_id: Company identifier
            use_semantic_search: If True, use vector search for relevance;
                                 if False, use all chunks for each year

        Returns:
            Tuple of (barriers_df, motivators_df)
        """
        print(f"\n{'='*60}")
        print(f"Extracting data for Company {company_id}")
        print(f"{'='*60}")

        years = self.get_years_for_company(company_id)
        print(f"Years found: {years}")

        barrier_rows = []
        motivator_rows = []

        for year in years:
            print(f"\nProcessing year {year}...")

            if use_semantic_search and self.retriever:
                barriers = self._extract_with_semantic_search(
                    company_id, year, "barriers"
                )
                motivators = self._extract_with_semantic_search(
                    company_id, year, "motivators"
                )
            else:
                barriers = self._extract_from_chunks(
                    company_id, year, "barriers"
                )
                motivators = self._extract_from_chunks(
                    company_id, year, "motivators"
                )

            barrier_rows.append({
                "company_id": company_id,
                "year": year,
                "barriers": barriers,
            })
            motivator_rows.append({
                "company_id": company_id,
                "year": year,
                "motivators": motivators,
            })

        return pd.DataFrame(barrier_rows), pd.DataFrame(motivator_rows)

    def _extract_with_semantic_search(
        self,
        company_id: str,
        year: str,
        extract_type: str
    ) -> str:
        """Extract using semantic search for relevant chunks."""
        query = BARRIER_QUERY if extract_type == "barriers" else MOTIVATOR_QUERY
        chain = self.barrier_chain if extract_type == "barriers" else self.motivator_chain
        empty_token = "NO_BARRIERS_FOUND" if extract_type == "barriers" else "NO_MOTIVATORS_FOUND"

        # Get all relevant docs
        all_docs = self.retriever.invoke(query)

        # Filter for company and year
        filtered_docs = [
            d for d in all_docs
            if d.metadata.get("company_id") == company_id
            and d.metadata.get("year") == year
        ]

        print(f"  → Found {len(filtered_docs)} relevant {extract_type} docs")

        if not filtered_docs:
            return empty_token

        # Use top N chunks
        context = "\n\n".join(
            d.page_content for d in filtered_docs[:self.config.context_chunks]
        )

        try:
            response = chain.invoke({"context": context})
            raw_text = response.content if hasattr(
                response, "content") else str(response)
            return self.clean_bullet_list(raw_text, empty_token=empty_token)
        except Exception as e:
            print(f"  ⚠️ Error extracting {extract_type}: {e}")
            return "ERROR"

    def _extract_from_chunks(
        self,
        company_id: str,
        year: str,
        extract_type: str
    ) -> str:
        """Extract using direct chunk access (no semantic search)."""
        chain = self.barrier_chain if extract_type == "barriers" else self.motivator_chain
        empty_token = "NO_BARRIERS_FOUND" if extract_type == "barriers" else "NO_MOTIVATORS_FOUND"

        # Get chunks for this company/year
        year_chunks = [
            c for c in self.chunks
            if c.metadata.get("company_id") == company_id
            and c.metadata.get("year") == year
        ]

        print(f"  → Found {len(year_chunks)} chunks for {extract_type}")

        if not year_chunks:
            return "NO_REPORT_FOR_YEAR"

        # Use first N chunks
        context = "\n\n".join(
            c.page_content for c in year_chunks[:self.config.context_chunks * 3]
        )

        try:
            response = chain.invoke({"context": context})
            raw_text = response.content if hasattr(
                response, "content") else str(response)
            return self.clean_bullet_list(raw_text, empty_token=empty_token)
        except Exception as e:
            print(f"  ⚠️ Error extracting {extract_type}: {e}")
            return "ERROR"

    def extract_all_companies(
        self,
        use_semantic_search: bool = True,
        save_results: bool = True
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Extract barriers and motivators for all companies.

        Returns:
            Dictionary mapping company_id to (barriers_df, motivators_df)
        """
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
        """
        Inspect loaded documents.

        Shows:
        - Total count
        - Sample documents with metadata
        - Company/year distribution
        """
        print(f"\n{'='*60}")
        print("DOCUMENT INSPECTION")
        print(f"{'='*60}")

        if not self.documents:
            print("⚠️ No documents loaded. Call load_pdfs() first.")
            return

        print(f"\nTotal documents: {len(self.documents)}")

        # Company distribution
        company_counts = {}
        for doc in self.documents:
            cid = doc.metadata.get("company_id", "Unknown")
            company_counts[cid] = company_counts.get(cid, 0) + 1

        print(f"\nDocuments per company:")
        for cid, count in sorted(company_counts.items()):
            print(f"  {cid}: {count}")

        # Sample documents
        print(f"\n--- Sample Documents (first {sample_size}) ---")
        for i, doc in enumerate(self.documents[:sample_size]):
            print(f"\n[Document {i+1}]")
            print(f"  Metadata: {doc.metadata}")
            print(f"  Content length: {len(doc.page_content)} chars")
            print(f"  Preview: {doc.page_content[:200]}...")

    def inspect_chunks(self, sample_size: int = 3):
        """
        Inspect chunked documents.

        Shows:
        - Total chunk count
        - Chunks per company/year
        - Sample chunks with metadata
        - Token distribution statistics
        """
        print(f"\n{'='*60}")
        print("CHUNK INSPECTION")
        print(f"{'='*60}")

        if not self.chunks:
            print("⚠️ No chunks created. Call chunk_documents() first.")
            return

        print(f"\nTotal chunks: {len(self.chunks)}")
        print(f"Chunk size setting: {self.config.chunk_size}")
        print(f"Chunk overlap setting: {self.config.chunk_overlap}")

        # Distribution by company
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

        # Token statistics
        print("\n--- Token Statistics ---")
        token_counts = []
        for chunk in tqdm(self.chunks[:100], desc="Counting tokens (sample)"):
            token_counts.append(self.count_tokens(chunk.page_content))

        print(f"  Min tokens: {min(token_counts)}")
        print(f"  Max tokens: {max(token_counts)}")
        print(f"  Avg tokens: {sum(token_counts)/len(token_counts):.1f}")
        print(
            f"  Over limit ({self.config.max_tokens}): {sum(1 for t in token_counts if t > self.config.max_tokens)}")

        # Sample chunks
        print(f"\n--- Sample Chunks (first {sample_size}) ---")
        for i, chunk in enumerate(self.chunks[:sample_size]):
            print(f"\n[Chunk {i+1}]")
            print(f"  ID: {chunk.metadata.get('chunk_id')}")
            print(f"  Company: {chunk.metadata.get('company_id')}")
            print(f"  Year: {chunk.metadata.get('year')}")
            print(f"  Tokens: {self.count_tokens(chunk.page_content)}")
            print(f"  Preview: {chunk.page_content[:150]}...")

    def inspect_vectorstore(self):
        """
        Inspect the vector store.

        Shows:
        - Total vectors
        - Index info
        - Sample similarity search
        """
        print(f"\n{'='*60}")
        print("VECTORSTORE INSPECTION")
        print(f"{'='*60}")

        if self.vectorstore is None:
            print(
                "⚠️ No vectorstore loaded. Call load_vectorstore() or embed_and_store().")
            return

        # Get index info
        index = self.vectorstore.index
        print(f"\nIndex type: {type(index).__name__}")
        print(f"Total vectors: {index.ntotal}")
        print(f"Vector dimension: {index.d}")

        # Check if we have a docstore
        if hasattr(self.vectorstore, 'docstore'):
            print(f"Docstore entries: {len(self.vectorstore.docstore._dict)}")

        # Test search
        print("\n--- Test Similarity Search ---")
        test_query = "carbon emissions reduction targets"
        results = self.vectorstore.similarity_search(test_query, k=3)

        print(f"Query: '{test_query}'")
        print(f"Results: {len(results)} documents")

        for i, doc in enumerate(results):
            print(f"\n  Result {i+1}:")
            print(f"    Company: {doc.metadata.get('company_id')}")
            print(f"    Year: {doc.metadata.get('year')}")
            print(f"    Preview: {doc.page_content[:100]}...")

    def inspect_extraction_sample(
        self,
        company_id: Optional[str] = None,
        year: Optional[str] = None
    ):
        """
        Run extraction on a single company/year and show detailed output.

        Useful for debugging extraction quality.
        """
        print(f"\n{'='*60}")
        print("EXTRACTION SAMPLE INSPECTION")
        print(f"{'='*60}")

        if not self.chunks:
            print("⚠️ No chunks loaded.")
            return

        # Default to first company/year if not specified
        if company_id is None:
            company_id = self.get_companies(
            )[0] if self.get_companies() else None

        if company_id is None:
            print("⚠️ No companies found.")
            return

        if year is None:
            years = self.get_years_for_company(company_id)
            year = years[0] if years else None

        if year is None:
            print(f"⚠️ No years found for company {company_id}.")
            return

        print(f"\nCompany: {company_id}")
        print(f"Year: {year}")

        # Get chunks for this company/year
        year_chunks = [
            c for c in self.chunks
            if c.metadata.get("company_id") == company_id
            and c.metadata.get("year") == year
        ]
        print(f"Available chunks: {len(year_chunks)}")

        # Show sample context
        context_sample = "\n\n".join(
            c.page_content[:300] for c in year_chunks[:3])
        print(f"\n--- Context Sample (first 3 chunks, truncated) ---")
        print(context_sample[:1000] + "...")

        # Run extraction
        print(f"\n--- Running Extraction ---")

        context_full = "\n\n".join(
            c.page_content for c in year_chunks[:self.config.context_chunks * 3]
        )

        print("\nBarriers extraction:")
        try:
            response = self.barrier_chain.invoke({"context": context_full})
            raw = response.content if hasattr(
                response, "content") else str(response)
            print(f"  Raw response:\n{raw}")
            cleaned = self.clean_bullet_list(raw, "NO_BARRIERS_FOUND")
            print(f"  Cleaned:\n{cleaned}")
        except Exception as e:
            print(f"  Error: {e}")

        print("\nMotivators extraction:")
        try:
            response = self.motivator_chain.invoke({"context": context_full})
            raw = response.content if hasattr(
                response, "content") else str(response)
            print(f"  Raw response:\n{raw}")
            cleaned = self.clean_bullet_list(raw, "NO_MOTIVATORS_FOUND")
            print(f"  Cleaned:\n{cleaned}")
        except Exception as e:
            print(f"  Error: {e}")

    def inspect_pipeline_status(self):
        """
        Show overall pipeline status and readiness.
        """
        print(f"\n{'='*60}")
        print("PIPELINE STATUS")
        print(f"{'='*60}")

        status = {
            "Documents loaded": len(self.documents) if self.documents else 0,
            "Chunks created": len(self.chunks) if self.chunks else 0,
            "Vectorstore": "✓ Ready" if self.vectorstore else "✗ Not loaded",
            "Retriever": "✓ Ready" if self.retriever else "✗ Not configured",
            "Retrieval chain": "✓ Ready" if self.retrieval_chain else "✗ Not configured",
            "LLM": "✓ Ready" if self._llm else "○ Not initialized (lazy)",
            "Embedding model": "✓ Ready" if self._embedding_model else "○ Not initialized (lazy)",
        }

        print("\nComponent Status:")
        for component, state in status.items():
            print(f"  {component}: {state}")

        print(f"\nConfiguration:")
        print(f"  Reports folder: {self.config.reports_folder}")
        print(f"  Vector DB path: {self.config.vector_db_path}")
        print(f"  Output folder: {self.config.output_folder}")
        print(f"  LLM model: {self.config.llm_model}")
        print(f"  Embedding model: {self.config.embedding_model_name}")

        if self.chunks:
            companies = self.get_companies()
            print(f"\nCompanies available: {companies}")

    def display_results(
        self,
        company_id: str,
        df_barriers: pd.DataFrame,
        df_motivators: pd.DataFrame
    ):
        """Display extraction results in a nice table format."""
        print(f"\n{'='*60}")
        print(f"BARRIERS - Company {company_id}")
        print(f"{'='*60}")
        print(tabulate(df_barriers, headers='keys', tablefmt='grid'))

        print(f"\n{'='*60}")
        print(f"MOTIVATORS - Company {company_id}")
        print(f"{'='*60}")
        print(tabulate(df_motivators, headers='keys', tablefmt='grid'))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_start(reports_folder: str = "../data/reports", incremental: bool = True):
    """
    Quick start function to get the pipeline running with minimal code.

    Usage:
        pipeline = quick_start("path/to/reports")
        pipeline.extract_all_companies()
    """
    pipeline = RAGPipeline()
    pipeline.config.reports_folder = reports_folder
    pipeline.load_and_process_pdfs(incremental=incremental)
    return pipeline


def load_existing_pipeline(vector_db_path: str = "vector_db/reports"):
    """
    Load an existing pipeline from a saved vector store.

    Usage:
        pipeline = load_existing_pipeline()
        pipeline.inspect_vectorstore()
    """
    config = PipelineConfig()
    config.vector_db_base = os.path.dirname(vector_db_path)
    config.vector_db_name = os.path.basename(vector_db_path)

    pipeline = RAGPipeline(config)
    pipeline.load_vectorstore()
    pipeline.setup_retrieval_chain()

    return pipeline


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("RAG Pipeline for Decarbonisation Report Analysis")
    print("=" * 60)
    print("\nUsage examples:")
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
    print("  # Load existing")
    print("  pipeline = load_existing_pipeline('vector_db/reports')")
