"""
BERTopic Modeling Pipeline
===========================

Clusters RAG-extracted barriers/motivators into coherent themes using BERTopic.

Pipeline:
  sentence embeddings → UMAP (compress) → HDBSCAN (cluster) → c-TF-IDF (keywords) → LLM labels

Key design choices:
- HDBSCAN over KMeans: no fixed topic count, identifies outliers, handles variable cluster sizes
- Embeddings cached separately from BERTopic fit: encoding is slow (~10s), re-fitting is fast (~2s),
  so cache_embeddings=True lets you tune HDBSCAN/UMAP without re-encoding
- Category-specific overrides: barriers and motivators have different cluster characteristics,
  so HDBSCAN/UMAP params are tuned per category via grid search (see topic_gridsearch.py)
- LLM topic labeling: batched into a single call (keywords → "N: Label" lines) to minimize API calls

Usage:
    from nlp import TopicModelConfig, run_topic_modeling_pipeline
    config = TopicModelConfig(embedding_model="ibm-granite/granite-embedding-english-r2")
    results = run_topic_modeling_pipeline(data_folder="../out", output_folder="../out/topics", config=config)
"""

from nlp import load_csv_data
from nlp import GPUManager
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, asdict, field, replace

import json
import numpy as np
import pandas as pd
# import datamapplot

from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()


# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Local imports


# ==================== Configuration ====================

@dataclass
class TopicModelConfig:
    """Configuration for topic modeling pipeline."""

    # Embedding
    embedding_model: str = "Snowflake/snowflake-arctic-embed-s"
    batch_size: int = 64  # embed (increase if GPU memory allows)
    # halves VRAM with negligible precision loss for inference; set to None for float32
    embedding_dtype: str = "bfloat16"
    # Encoding is slow (~10s); BERTopic refit is fast (~2s). True = skip re-encoding on rerun.
    cache_embeddings: bool = True

    # misc
    verbose: bool = True
    # base prefix for auto-increment (run_01, run_02,...)
    run_name: str = "run"

    # UMAP parameters
    umap_n_neighbors: int = 15
    umap_n_components: int = 30
    umap_min_dist: float = 0.05  # higher = worse cluster, but better viz
    umap_metric: str = 'cosine'  # 'cosine'
    umap_random_state: int = 42

    # HDBSCAN parameters
    # min_cluster_size controls number of topics (higher = fewer topics)
    hdbscan_min_cluster_size: int = 20  # ~1% of dataset
    hdbscan_min_samples: int = 3  # Lower = less noise/outliers
    hdbscan_metric: str = 'euclidean'
    hdbscan_cluster_selection_method: str = 'eom'  # 'eom' or 'leaf'

    vectorizer_ngram_range: Tuple[int, int] = (1, 2)  # Include bigrams
    # Minimum document frequency (use 1 for small topics, 2+ for large datasets)
    vectorizer_min_df: int = 2
    vectorizer_max_df: float = 0.92  # rm common >92%

    mmr_diversity: float = 0.4  # 0=pure relevance, 1=max diversity # !!

    # BERTopic parameters
    top_n_words: int = 10
    # let HDBSCAN find natural clusters with larger min_cluster_size
    nr_topics: Optional[int] = None
    # Set True for soft clustering (slower, not used yet)
    calculate_probabilities: bool = False
    # Reduce outliers by assigning to nearest topic (post-hoc)
    reduce_outliers: bool = True
    # 'embeddings' (nearest neighbor in embed space) or 'c-tf-idf' (keyword overlap) or 'distributions'
    # 'embeddings' is the most semantically faithful strategy
    reduce_outliers_strategy: str = "embeddings"

    # Visualization
    # 2D UMAP for visualization (separate from clustering)
    viz_umap_n_neighbors: int = 5
    viz_umap_n_components: int = 2
    viz_umap_min_dist: float = 0.0

    # LLM settings for topic labeling
    llm_provider: str = "groq"  # "ollama"
    model: str = "llama-3.1-8b-instant"

    ollama_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.0

    # Per-category HDBSCAN/UMAP overrides from grid search — barriers and motivators
    # cluster differently so use separate params. Set via TopicGridSearch or manually.
    # E.g. {"barriers": {"hdbscan_min_cluster_size": 25, "umap_n_components": 5}}
    category_overrides: Optional[Dict[str, Dict[str, Any]]] = field(
        default_factory=dict)


# ==================== Pipeline State ====================

@dataclass
class PipelineState:
    """Cross-session state for the notebook pipeline, persisted to JSON.

    Wraps the static config (model names, batch size) and dynamic derived
    values (run_dir, category_overrides from grid search). On kernel restart,
    call load_or_create() once and all subsequent cells reuse config + RUN_DIR
    without re-initializing or copy-pasting.

    kwargs to load_or_create() always win over saved values, so the notebook
    cell is the source of truth for config while the JSON file persists
    derived state (run_dir, category_overrides) across sessions.

    Usage:
        state = PipelineState.load_or_create("../out/topics/state.json",
                    embedding_model="ibm-granite/granite-embedding-english-r2",
                    llm_provider="groq", llm_model="llama-3.1-8b-instant")
        config = state.config   # TopicModelConfig built from current state
        RUN_DIR = state.run_dir # None if never run, otherwise last run path
    """
    state_file: str

    # Static config — set in notebook, persisted for convenience
    embedding_model: str = "ibm-granite/granite-embedding-english-r2"
    batch_size: int = 64
    llm_provider: str = "groq"
    llm_model: str = "llama-3.1-8b-instant"

    # Dynamic state — derived by pipeline steps, persisted across sessions
    run_dir: Optional[str] = None
    category_overrides: Optional[Dict[str, Dict[str, Any]]] = None

    @classmethod
    def load_or_create(cls, state_file: str, **kwargs) -> "PipelineState":
        """Load from JSON if it exists; kwargs override any saved values."""
        saved = {}
        if os.path.exists(state_file):
            with open(state_file) as f:
                saved = json.load(f)
            print(f"📂 Loaded state from {state_file}")
        merged = {**saved, **kwargs}
        valid = {k: v for k, v in merged.items() if k in cls.__dataclass_fields__}
        return cls(state_file=state_file, **valid)

    def save(self):
        """Persist current state to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(self.state_file)), exist_ok=True)
        data = {k: v for k, v in asdict(self).items() if k != "state_file"}
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"💾 State saved → {self.state_file}")

    @property
    def config(self) -> "TopicModelConfig":
        """Build TopicModelConfig from current state."""
        return TopicModelConfig(
            embedding_model=self.embedding_model,
            batch_size=self.batch_size,
            llm_provider=self.llm_provider,
            model=self.llm_model,
            category_overrides=self.category_overrides or {},
        )

    def __repr__(self) -> str:
        lines = [f"PipelineState  ({self.state_file})"]
        lines.append(f"  embedding    : {self.embedding_model}")
        lines.append(f"  llm          : {self.llm_provider} / {self.llm_model}")
        lines.append(f"  run_dir      : {self.run_dir or '(not set)'}")
        if self.category_overrides:
            for cat, params in self.category_overrides.items():
                lines.append(f"  overrides.{cat:<12}: {params}")
        else:
            lines.append("  overrides    : (not set — run grid search or set manually)")
        return "\n".join(lines)


# ==================== Prompt ====================

TOPIC_LABEL_PROMPT = ChatPromptTemplate.from_template("""You are an expert analyst specializing in thematic classification and topic labeling.

Below are numbered topics, each with keywords. Generate a concise label for EACH topic.

{keywords}

OUTPUT FORMAT (one per line, same numbering):
1: Label Here
2: Another Label

LABEL REQUIREMENTS:
- Length: 3–5 words, Title Case
- Use & instead of "and" where appropriate
- Describe a specific barrier or motivator
- Name a concrete issue using a clear noun phrase

DO NOT:
- Explain reasoning or add comments
- Mention keywords or documents
- Use meta terms like Theme, Topic, or Issue
- Add quotes, punctuation, or extra text

GOOD: Raw Materials & Energy Availability, Fossil-Free Steel Innovation
BAD: Operations, Cost, Various challenges in the steel production process

Return ONLY the numbered labels, one per line.
""")


# ==================== Keyword Filtering ====================

# Company names and other terms to filter from keywords before LLM labeling
KEYWORD_STOPWORDS = {
    # Steel companies
    "outokumpu", "ssab", "baosteel", "arcelormittal", "thyssenkrupp",
    "voestalpine", "tata", "nippon", "posco", "jfe", "nucor", "salzgitter",
    "aperam", "acerinox", "nlmk", "severstal", "evraz",
    # Generic terms that don't help labeling
    "company", "group", "companies", "report", "annual", "year", "years",
    "million", "billion", "EUR", "USD", "barriers", "barrier", "risk", "risks",
    "mention", "mentioned", "mentions",
    "qualifying", "motivators", "motivating", "motivator",
    # Filler words that sometimes appear
    "also", "including", "various", "related", "based", "using",
    # Domain-generic filler terms
    "use", "used", "need", "process", "increase", "help", "end", "example",
    "implement", "impact", "product", "products", "measures", "term", "meet",
    "support", "ways", "sources", "methods", "approach", "changes", "required",
    "available", "information", "certain", "brand", "applications",
    "requirements", "structure", "operations", "statement",
    "sludge", "wastewater", "sludge waste", "plastics",
    "esrs", "sfdr", "pillar", "anti-corruption", "gri"
}


def _filter_keywords(keywords: str) -> str:
    """Filter out company names and generic terms from keywords.

    Args:
        keywords: Comma-separated keyword string

    Returns:
        Filtered comma-separated keyword string
    """
    # TODO: can company names slip?
    words = [w.strip() for w in keywords.split(",")]
    filtered = [
        w for w in words
        if w.lower() not in KEYWORD_STOPWORDS
        and not any(stop in w.lower() for stop in KEYWORD_STOPWORDS)
    ]
    return ", ".join(filtered)


# ==================== Topic Modeler ====================

class TopicModeler:
    """
    BERTopic-based topic modeling with HDBSCAN clustering.

    Uses:
    - UMAP for dimensionality reduction
    - HDBSCAN for density-based clustering (no fixed cluster count)
    - KeyBERTInspired for better topic word representation
    - DataMapPlot for visualization (optional)

    Example:
        modeler = TopicModeler()
        df = modeler.fit_transform(df, text_column='barriers')
        modeler.visualize_topics()
        modeler.visualize_documents(df['barriers'].tolist())
    """

    def __init__(self, config: Optional[TopicModelConfig] = None):
        """
        Initialize TopicModeler.

        Args:
            config: TopicModelConfig with model parameters
        """
        self.config = config or TopicModelConfig()
        self.gpu = GPUManager()

        # Models (initialized lazily)
        self._embedding_model = None
        self._topic_model = None
        self._embeddings = None
        self._reduced_embeddings = None
        self._llm = None

        self._log(f"🖥️  {self.gpu}")

    def _log(self, message: str):
        """Print message if verbose mode enabled."""
        if self.config.verbose:
            print(message)

    @property
    def embedding_model(self):
        """Lazy-load embedding model."""
        if self._embedding_model is None:
            self._load_embedding_model()
        return self._embedding_model

    @property
    def topic_model(self):
        """Get fitted topic model."""
        if self._topic_model is None:
            raise ValueError(
                "Topic model not fitted. Call fit_transform() first.")
        return self._topic_model

    @property
    def llm(self):
        """Lazy-load the LLM (Ollama or Groq)."""
        if self._llm is None:
            if self.config.llm_provider == "groq":
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError(
                        "GROQ_API_KEY not found. Check .env file.")
                self._log(f"Loading Groq: {self.config.model}")
                self._llm = ChatGroq(
                    model=self.config.model, api_key=api_key, temperature=self.config.llm_temperature)
            else:
                self._log(f"Loading Ollama: {self.config.model}")
                self._llm = ChatOllama(
                    model=self.config.model, base_url=self.config.ollama_base_url, temperature=self.config.llm_temperature)
        return self._llm

    def _load_embedding_model(self):
        """Load sentence transformer embedding model."""

        dtype_str = self.config.embedding_dtype
        self._log(f"\n🤖 Loading embedding model: {self.config.embedding_model}"
                  + (f" ({dtype_str})" if dtype_str else ""))
        model_kwargs = {"torch_dtype": dtype_str} if dtype_str else {}
        self._embedding_model = SentenceTransformer(
            self.config.embedding_model, model_kwargs=model_kwargs)

        # Move to GPU if available
        if self.gpu.is_cuda:
            self._embedding_model = self._embedding_model.to(self.gpu.device)
            self._log("✅ Using GPU for encoding")
        else:
            self._log("⚠️  Using CPU for encoding (slower)")

    def _create_topic_model(self, embedding_model=None):
        """Create BERTopic model with configured components."""
        self._log("\n🔧 Creating BERTopic model components...")

        # Step 1: Dimensionality reduction with UMAP
        umap_model = UMAP(
            n_neighbors=self.config.umap_n_neighbors,
            n_components=self.config.umap_n_components,
            min_dist=self.config.umap_min_dist,
            metric=self.config.umap_metric,
            random_state=self.config.umap_random_state
        )
        self._log(f"  📐 UMAP: {self.config.umap_n_components}D, "
                  f"n_neighbors={self.config.umap_n_neighbors}")

        # Step 2: Clustering with HDBSCAN
        # HDBSCAN advantages over KMeans:
        # - No need to specify number of clusters
        # - Identifies outliers (topic -1)
        # - Finds clusters of varying sizes and shapes
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.config.hdbscan_min_cluster_size,
            min_samples=self.config.hdbscan_min_samples,
            metric=self.config.hdbscan_metric,
            cluster_selection_method=self.config.hdbscan_cluster_selection_method,
            prediction_data=True,  # Required for BERTopic
            gen_min_span_tree=True  # Can improve clusters
        )
        self._log(f"  🎯 HDBSCAN: min_cluster_size={self.config.hdbscan_min_cluster_size}, "
                  f"min_samples={self.config.hdbscan_min_samples}")

        # Step 3: Vectorizer for c-TF-IDF
        # Combine English stop words with custom domain stopwords (company names, generic terms)
        combined_stop_words = list(ENGLISH_STOP_WORDS | KEYWORD_STOPWORDS)
        vectorizer_model = CountVectorizer(
            stop_words=combined_stop_words,
            ngram_range=self.config.vectorizer_ngram_range,
            min_df=self.config.vectorizer_min_df,
            max_df=self.config.vectorizer_max_df
        )

        # Step 4: c-TF-IDF transformer
        ctfidf_model = ClassTfidfTransformer()

        # Step 5: Representation models
        # KeyBERTInspired: refines keywords using embedding similarity
        # MMR: adds diversity to reduce redundant keywords
        keybert_model = KeyBERTInspired(
            top_n_words=self.config.top_n_words
        )
        mmr_model = MaximalMarginalRelevance(
            diversity=self.config.mmr_diversity
        )
        representation_model = [keybert_model, mmr_model]
        self._log(
            f"  🔑 Using KeyBERTInspired + MMR (diversity={self.config.mmr_diversity})")

        # Create BERTopic model
        topic_model = BERTopic(
            # Pipeline components
            embedding_model=embedding_model or self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=representation_model,
            # Parameters
            top_n_words=self.config.top_n_words,
            nr_topics=self.config.nr_topics,
            calculate_probabilities=self.config.calculate_probabilities,
            verbose=self.config.verbose
        )

        return topic_model

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for documents.

        Args:
            texts: List of text documents

        Returns:
            numpy array of embeddings
        """
        self._log(f"\n🧮 Generating embeddings for {len(texts)} documents...")

        embeddings = self.embedding_model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=self.config.verbose,
            convert_to_numpy=True,
            normalize_embeddings=True  # for BGE embed
        )

        return embeddings

    def fit_transform(
        self,
        df: pd.DataFrame,
        text_column: str,
        embeddings: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, List[int], Optional[np.ndarray]]:
        """
        Fit topic model and transform documents.

        Args:
            df: DataFrame with text data
            text_column: Name of column containing text
            embeddings: Pre-computed embeddings (optional)

        Returns:
            Tuple of (DataFrame with topics, topics list, probabilities)
        """
        texts = df[text_column].tolist()
        self._log(
            f"\n🔄 Processing {len(texts)} documents from '{text_column}'...")

        # Generate embeddings if not provided
        embeddings_precomputed = embeddings is not None
        if not embeddings_precomputed:
            embeddings = self.encode_documents(texts)

        self._embeddings = embeddings

        # Pass string name when embeddings are pre-computed to skip eager model load
        self._topic_model = self._create_topic_model(
            embedding_model=self.config.embedding_model if embeddings_precomputed else None)

        self._log("\n📊 Fitting BERTopic model...")
        topics, probs = self._topic_model.fit_transform(texts, embeddings)

        # Add topics to dataframe
        df = df.copy()
        df['topic'] = topics

        # Get topic info
        topic_info = self._topic_model.get_topic_info()
        n_topics = len(topic_info[topic_info['Topic'] != -1])
        n_outliers = (np.array(topics) == -1).sum()

        self._log(f"\n✅ Found {n_topics} topics")
        self._log(
            f"   📍 {n_outliers} documents classified as outliers (topic -1)")

        return df, topics, probs

    def get_topic_info(self) -> pd.DataFrame:
        """Get topic information."""
        return self.topic_model.get_topic_info()

    def get_topic(self, topic_id: int, full: bool = False) -> Dict:
        """
        Get topic representation.

        Args:
            topic_id: Topic ID
            full: If True, return all representations (Main, KeyBERT, etc.)
        """
        return self.topic_model.get_topic(topic_id, full=full)

    def print_topics(self, n_topics: Optional[int] = None):
        """Print topic summaries."""
        topic_info = self.get_topic_info()

        # Exclude outlier topic (-1)
        topics = topic_info
        if n_topics:
            topics = topics.head(n_topics)

        self._log("\n📋 Topic Summary:")
        self._log("-" * 60)

        for _, row in topics.iterrows():
            topic_id = row['Topic']
            count = row['Count']
            name = row.get('Name', f"Topic_{topic_id}")

            # Get top words
            top_words = self.topic_model.get_topic(topic_id)
            if top_words:
                words = [word for word, _ in top_words[:5]]
                words_str = ", ".join(words)
            else:
                words_str = "N/A"

            self._log(f"Topic {topic_id}: {count:4d} docs | {words_str}")

        # Show outliers
        outliers = topic_info[topic_info['Topic'] == -1]
        if not outliers.empty:
            self._log(
                f"\nOutliers (Topic -1): {outliers['Count'].values[0]} docs")

    def generate_topic_labels(self) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, int]]:
        """
        Generate human-readable labels for topics using a single batched LLM call.

        Returns:
            Tuple of (labels_dict, keywords_dict, doc_count_dict)
        """
        topic_info = self.get_topic_info()
        labels = {}
        keywords_map = {}
        doc_counts = {}

        # Collect keywords for all non-outlier topics
        batch_lines = []  # (line_number, topic_id)
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            doc_counts[topic_id] = row['Count']

            top_words = self.topic_model.get_topic(topic_id)
            keywords_raw = ", ".join(
                [w for w, _ in top_words[:10]]) if top_words else ""
            keywords_map[topic_id] = keywords_raw

            if topic_id == -1:
                labels[topic_id] = "Outliers (unassigned)"
                continue

            if not top_words:
                labels[topic_id] = f"Topic {topic_id}"
                continue

            keywords_filtered = _filter_keywords(keywords_raw)
            line_num = len(batch_lines) + 1
            batch_lines.append((line_num, topic_id))
            labels[topic_id] = f"Topic {topic_id}"  # fallback

        if not batch_lines:
            return labels, keywords_map, doc_counts

        # Build numbered keyword list
        keyword_block = "\n".join(
            f"{ln}: {_filter_keywords(keywords_map[tid])}" for ln, tid in batch_lines
        )

        self._log(
            f"\n🏷️  Generating {len(batch_lines)} topic labels with LLM (single batch)...")
        chain = TOPIC_LABEL_PROMPT | self.llm

        try:
            response = chain.invoke({"keywords": keyword_block})
            text = response.content if hasattr(
                response, "content") else str(response)

            # Parse "1: Label" lines
            parsed = {}
            for line in text.strip().splitlines():
                line = line.strip()
                if ":" in line:
                    num_str, label = line.split(":", 1)
                    num_str = num_str.strip().rstrip(".")
                    if num_str.isdigit():
                        parsed[int(num_str)] = label.strip().strip('"\'')

            # Map back to topic IDs
            for line_num, topic_id in batch_lines:
                if line_num in parsed:
                    labels[topic_id] = parsed[line_num]
                    self._log(f"  Topic {topic_id}: {labels[topic_id]}")
                else:
                    self._log(
                        f"  ⚠️ Topic {topic_id}: no label parsed, using fallback")

        except Exception as e:
            self._log(f"  ⚠️ Batch labeling failed: {e}")

        return labels, keywords_map, doc_counts

    def set_topic_labels(self, labels: Dict[int, str]):
        """
        Set custom labels for topics.

        Args:
            labels: Dict mapping topic_id -> label string
        """
        self.topic_model.set_topic_labels(labels)
        self._log(f"✅ Set labels for {len(labels)} topics")

    def reduce_outliers(
        self,
        texts: List[str],
        topics: List[int],
        strategy: str = "embeddings"
    ) -> List[int]:
        """
        Reduce outliers by assigning them to nearest topics.

        Args:
            texts: Original documents
            topics: Current topic assignments
            strategy: 'embeddings' or 'distributions'

        Returns:
            Updated topic assignments
        """
        # Check if there are any outliers to reduce
        n_outliers = sum(1 for t in topics if t == -1)
        if n_outliers == 0:
            self._log("✅ No outliers to reduce (0 documents in topic -1)")
            return topics

        self._log(
            f"\n🔄 Reducing {n_outliers} outliers using '{strategy}' strategy...")

        if strategy == "embeddings" and self._embeddings is not None:
            new_topics = self.topic_model.reduce_outliers(
                texts, topics, strategy="embeddings", embeddings=self._embeddings
            )
        else:
            new_topics = self.topic_model.reduce_outliers(texts, topics)

        n_reduced = sum(1 for old, new in zip(
            topics, new_topics) if old == -1 and new != -1)
        self._log(f"✅ Reassigned {n_reduced} outliers to topics")

        return new_topics

    def reduce_embeddings_for_viz(
        self,
        embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Reduce embeddings to 2D for visualization.

        Args:
            embeddings: Embeddings to reduce (uses stored if None)

        Returns:
            2D embeddings
        """

        if embeddings is None:
            embeddings = self._embeddings

        if embeddings is None:
            raise ValueError(
                "No embeddings available. Call fit_transform() first.")

        self._log("\n📉 Reducing embeddings to 2D for visualization...")

        umap_2d = UMAP(
            n_neighbors=self.config.viz_umap_n_neighbors,
            n_components=self.config.viz_umap_n_components,
            min_dist=self.config.viz_umap_min_dist,
            metric=self.config.umap_metric,
            random_state=self.config.umap_random_state
        )

        self._reduced_embeddings = umap_2d.fit_transform(embeddings)
        return self._reduced_embeddings

    # ==================== Visualization ====================

    def viz(
        self,
        df: pd.DataFrame,
        text_column: str,
        output_path: str,
        category: str,
        top_n_topics: int = 3,
        topics: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Generate and save all visualizations for a category.

        Args:
            df: DataFrame with text data and metadata (year, company columns)
            text_column: Name of column containing document text
            output_path: Directory to save visualizations
            category: Category name ("barriers" or "motivators")
            top_n_topics: Number of topics for barchart (ignored when topics is set)
            topics: Explicit list of topic IDs to show — filters all viz to these topics.
                    When None, all topics are shown.

        Returns:
            Dict with paths to saved files and any errors
        """
        output = Path(output_path)
        output.mkdir(parents=True, exist_ok=True)

        self._log(f"\n📊 Generating visualizations for {category}...")
        if topics:
            self._log(f"   🔍 Filtering to {len(topics)} topics: {topics}")

        results = {"saved": [], "errors": []}

        # Ensure 2D embeddings are computed for document visualizations
        if self._reduced_embeddings is None and self._embeddings is not None:
            self.reduce_embeddings_for_viz()

        # When filtering to specific topics, restrict docs and embeddings too
        if topics is not None:
            mask = df["topic"].isin(topics).values
            docs = df[mask][text_column].tolist()
            emb = self._embeddings[mask] if self._embeddings is not None else None
            red_emb = self._reduced_embeddings[mask] if self._reduced_embeddings is not None else None
        else:
            docs = df[text_column].tolist()
            emb = self._embeddings
            red_emb = self._reduced_embeddings

        viz_configs = [
            ("barchart", lambda: self.topic_model.visualize_barchart(
                topics=topics, top_n_topics=top_n_topics)),
            ("topics_2d", lambda: self.topic_model.visualize_topics(
                topics=topics)),
            ("hierarchy", lambda: self.topic_model.visualize_hierarchy(
                topics=topics)),
            ("heatmap", lambda: self.topic_model.visualize_heatmap(
                topics=topics)),
            ("documents", lambda: self.topic_model.visualize_documents(
                docs,
                topics=topics,
                embeddings=emb,
                reduced_embeddings=red_emb,
            )),
        ]

        for name, viz_func in viz_configs:
            try:
                fig = viz_func()
                path = output / f"{category}_{name}.html"
                fig.write_html(str(path))
                results["saved"].append(str(path))
                self._log(f"  ✓ {category}_{name}.html")
            except Exception as e:
                results["errors"].append((name, str(e)))
                self._log(f"  ⚠️ {name}: {e}")

        # Topics over time (using year column)
        if 'year' in df.columns:
            try:
                years = df[df["topic"].isin(topics)]["year"].tolist() if topics is not None else df['year'].tolist()
                topics_over_time = self.topic_model.topics_over_time(
                    docs, years,
                    global_tuning=True,
                    evolution_tuning=True
                )
                fig = self.topic_model.visualize_topics_over_time(
                    topics_over_time,
                    topics=topics,
                    top_n_topics=top_n_topics,
                )
                path = output / f"{category}_over_time.html"
                fig.write_html(str(path))
                results["saved"].append(str(path))
                self._log(f"  ✓ {category}_over_time.html")
            except Exception as e:
                results["errors"].append(("over_time", str(e)))
                self._log(f"  ⚠️ over_time: {e}")

        # Topics per company
        if 'company' in df.columns:
            try:
                df_viz = df[df["topic"].isin(topics)] if topics is not None else df
                companies = df_viz['company'].tolist()
                topics_per_company = self.topic_model.topics_per_class(
                    docs, companies)
                fig = self.topic_model.visualize_topics_per_class(
                    topics_per_company,
                    topics=topics,
                    top_n_topics=top_n_topics,
                )
                path = output / f"{category}_per_company.html"
                fig.write_html(str(path))
                results["saved"].append(str(path))
                self._log(f"  ✓ {category}_per_company.html")
            except Exception as e:
                results["errors"].append(("per_company", str(e)))
                self._log(f"  ⚠️ per_company: {e}")

        # DataMapPlot (requires datamapplot package)
        try:
            fig = self.topic_model.visualize_document_datamap(
                docs,
                embeddings=emb,
                reduced_embeddings=red_emb,
                title=f"{category.title()} Topics"
            )
            path = output / f"{category}_datamap.png"
            fig.savefig(str(path), dpi=150, bbox_inches='tight')
            results["saved"].append(str(path))
            self._log(f"  ✓ {category}_datamap.png")
        except Exception as e:
            results["errors"].append(("datamap", str(e)))
            self._log(f"  ⚠️ datamap: {e}")

        return results

    # ==================== Persistence ====================

    def save(self, path: str, save_embeddings: bool = True):
        """
        Save topic model and optionally embeddings.

        Args:
            path: Path to save model (without extension)
            save_embeddings: Whether to save embeddings
        """
        self.topic_model.save(f"{path}_model")
        self._log(f"✅ Saved model to {path}_model")

        if save_embeddings and self._embeddings is not None:
            np.save(f"{path}_embeddings.npy", self._embeddings)
            self._log(f"✅ Saved embeddings to {path}_embeddings.npy")

        if self._reduced_embeddings is not None:
            np.save(f"{path}_reduced_embeddings.npy", self._reduced_embeddings)

    def load(self, path: str, load_embeddings: bool = True):
        """
        Load topic model and optionally embeddings.

        Args:
            path: Path to saved model (without extension)
            load_embeddings: Whether to load embeddings
        """

        self._topic_model = BERTopic.load(f"{path}_model")
        self._log(f"✅ Loaded model from {path}_model")

        if load_embeddings:
            emb_path = f"{path}_embeddings.npy"
            if os.path.exists(emb_path):
                self._embeddings = np.load(emb_path)
                self._log(f"✅ Loaded embeddings from {emb_path}")

            red_emb_path = f"{path}_reduced_embeddings.npy"
            if os.path.exists(red_emb_path):
                self._reduced_embeddings = np.load(red_emb_path)

    def cleanup(self):
        """Clean up GPU memory."""
        self.gpu.clear()


# ==================== Aggregation Functions ====================

def aggregate_by_year(
    df: pd.DataFrame,
    topic_column: str = 'topic',
    year_column: str = 'year'
) -> pd.DataFrame:
    """
    Aggregate topic counts by year.

    Args:
        df: DataFrame with topic and year columns
        topic_column: Name of topic column
        year_column: Name of year column

    Returns:
        Pivoted DataFrame with years as rows and topics as columns
    """
    yearly_counts = df.groupby(
        [year_column, topic_column]).size().reset_index(name='count')
    return yearly_counts.pivot(
        index=year_column,
        columns=topic_column,
        values='count'
    ).fillna(0).astype(int)


def aggregate_by_category(
    df: pd.DataFrame,
    category_column: str,
    topic_column: str = 'topic'
) -> pd.DataFrame:
    """
    Aggregate topic counts by category.

    Args:
        df: DataFrame with topic and category columns
        category_column: Name of category column
        topic_column: Name of topic column

    Returns:
        Pivoted DataFrame with categories as rows and topics as columns
    """
    counts = df.groupby([category_column, topic_column]
                        ).size().reset_index(name='count')
    return counts.pivot(
        index=category_column,
        columns=topic_column,
        values='count'
    ).fillna(0).astype(int)


def aggregate_by_company_year(
    df: pd.DataFrame,
    topic_column: str = 'topic',
    company_column: str = 'company',
    year_column: str = 'year'
) -> pd.DataFrame:
    """
    Aggregate topic counts by company AND year.

    Creates a pivot table showing topic distribution across company-year combinations,
    useful for tracking how topics evolve within each company over time.

    Args:
        df: DataFrame with topic, company, and year columns
        topic_column: Name of topic column
        company_column: Name of company column
        year_column: Name of year column

    Returns:
        Pivoted DataFrame with (company, year) as multi-index rows and topics as columns
    """
    counts = df.groupby(
        [company_column, year_column, topic_column]
    ).size().reset_index(name='count')

    return counts.pivot_table(
        index=[company_column, year_column],
        columns=topic_column,
        values='count',
        fill_value=0
    ).astype(int)


# ==================== Grid Search ====================

def run_grid_search(
    data_folder: str,
    output_folder: str = "./output",
    category: str = "barriers",
    config: Optional[TopicModelConfig] = None,
    param_grid: Optional[Dict[str, List]] = None,
) -> pd.DataFrame:
    """
    Sweep HDBSCAN/UMAP param combos using cached embeddings (~2s each).

    Args:
        data_folder: Path to folder with CSV data files
        output_folder: Path where embedding caches live (same as pipeline output)
        category: "barriers" or "motivators"
        config: Base TopicModelConfig (uses defaults if None)
        param_grid: Dict of param_name -> list of values to try.
            Defaults to a sweep over hdbscan_min_cluster_size and hdbscan_min_samples.

    Returns:
        DataFrame with one row per combo: param columns + n_topics, n_outliers,
        outlier_pct, topic_size_min/max/median/mean, largest_topic_pct
    """
    from itertools import product

    config = config or TopicModelConfig()

    if param_grid is None:
        param_grid = {
            "hdbscan_min_cluster_size": [8, 12, 16, 20, 25, 30, 40],
            "hdbscan_min_samples": [2, 3, 5],
        }

    # Load data
    df = load_csv_data(data_folder, category)
    if df.empty:
        raise ValueError(f"No {category} data found in {data_folder}")
    texts = df[category].tolist()

    # Load cached embeddings (keyed by model name to avoid stale cache)
    model_slug = config.embedding_model.split("/")[-1]
    embed_file = Path(output_folder) / \
        f"embeddings_{category}_{model_slug}.npy"
    if not embed_file.exists():
        # Fall back to legacy filename (before model-keyed caching)
        legacy = Path(output_folder) / f"embeddings_{category}.npy"
        if legacy.exists():
            embed_file = legacy
        else:
            raise FileNotFoundError(
                f"No cached embeddings at {embed_file}. Run the pipeline once first "
                f"with cache_embeddings=True to generate them."
            )
    embeddings = np.load(str(embed_file))
    if len(embeddings) != len(df):
        raise ValueError(
            f"Embedding cache size ({len(embeddings)}) != data size ({len(df)})")
    print(
        f"✓ Loaded {len(texts)} docs + cached embeddings for '{category}' ({embed_file.name})")

    # Build all combos
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combos = list(product(*param_values))
    print(f"🔍 Grid search: {len(combos)} combos over {param_names}")

    rows = []
    for i, combo in enumerate(combos):
        overrides = dict(zip(param_names, combo))
        trial_config = replace(config, **overrides)
        modeler = TopicModeler(trial_config)
        modeler.config.verbose = False

        try:
            _, topics, _ = modeler.fit_transform(
                df, category, embeddings=embeddings)

            topic_arr = np.array(topics)
            n_outliers = int((topic_arr == -1).sum())
            n_total = len(topic_arr)
            non_outlier = topic_arr[topic_arr != -1]

            # Topic size stats (excluding outliers)
            topic_info = modeler.get_topic_info()
            sizes = topic_info[topic_info["Topic"] != -1]["Count"].values
            n_topics = len(sizes)

            # DBCV — intrinsic cluster validity from HDBSCAN (higher = better-separated clusters)
            hdbscan_model = modeler.topic_model.hdbscan_model
            dbcv = round(float(hdbscan_model.relative_validity_), 4) if hasattr(
                hdbscan_model, "relative_validity_") else None

            row = {**overrides}
            row["n_topics"] = n_topics
            row["n_outliers"] = n_outliers
            row["outlier_pct"] = round(n_outliers / n_total * 100, 1)
            row["dbcv"] = dbcv
            row["topic_size_min"] = int(sizes.min()) if len(sizes) else 0
            row["topic_size_max"] = int(sizes.max()) if len(sizes) else 0
            row["topic_size_median"] = int(
                np.median(sizes)) if len(sizes) else 0
            row["topic_size_mean"] = round(
                float(sizes.mean()), 1) if len(sizes) else 0
            row["largest_topic_pct"] = round(
                int(sizes.max()) / n_total * 100, 1) if len(sizes) else 0

            status = "✓"
        except Exception as e:
            row = {**overrides, "n_topics": 0, "n_outliers": 0, "outlier_pct": 0, "dbcv": None,
                   "topic_size_min": 0, "topic_size_max": 0, "topic_size_median": 0,
                   "topic_size_mean": 0, "largest_topic_pct": 0, "error": str(e)}
            status = "✗"

        rows.append(row)
        print(
            f"  {status} [{i+1}/{len(combos)}] {overrides} → {row.get('n_topics', '?')} topics, {row.get('outlier_pct', '?')}% outliers")

        modeler.cleanup()

    result_df = pd.DataFrame(rows)
    print(f"\n✅ Grid search complete: {len(result_df)} results")
    return result_df


# ==================== Main Pipeline ====================

def _write_config_log(output_path: Path, config: TopicModelConfig, duration_s: float = 0.0, results: Optional[Dict[str, Any]] = None):
    """Write config parameters to a log file for reproducibility."""
    log_file = output_path / "config_log.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"{'='*60}",
        f"Topic Modeling Pipeline Run",
        f"Timestamp: {timestamp}",
        f"Duration: {duration_s:.1f}s",
        f"{'='*60}",
    ]

    def section(title, params):
        lines.append("")
        lines.append(f"{title}:")
        for k, v in params:
            lines.append(f"  {k}: {v}")

    section("Embedding", [
        ("model", config.embedding_model),
        ("batch_size", config.batch_size),
    ])
    section("UMAP (clustering)", [
        ("n_neighbors", config.umap_n_neighbors),
        ("n_components", config.umap_n_components),
        ("min_dist", config.umap_min_dist),
        ("metric", config.umap_metric),
        ("random_state", config.umap_random_state),
    ])
    section("HDBSCAN", [
        ("min_cluster_size", config.hdbscan_min_cluster_size),
        ("min_samples", config.hdbscan_min_samples),
        ("metric", config.hdbscan_metric),
        ("cluster_selection_method", config.hdbscan_cluster_selection_method),
    ])
    section("Vectorizer", [
        ("ngram_range", config.vectorizer_ngram_range),
        ("min_df", config.vectorizer_min_df),
        ("max_df", config.vectorizer_max_df),
    ])
    section("Representation", [
        ("mmr_diversity", config.mmr_diversity),
        ("top_n_words", config.top_n_words),
    ])
    section("BERTopic", [
        ("nr_topics", config.nr_topics),
        ("reduce_outliers",
         f"{config.reduce_outliers} ({config.reduce_outliers_strategy})"),
        ("calculate_probabilities", config.calculate_probabilities),
    ])
    section("Visualization UMAP", [
        ("n_neighbors", config.viz_umap_n_neighbors),
        ("n_components", config.viz_umap_n_components),
        ("min_dist", config.viz_umap_min_dist),
    ])
    section("LLM", [
        ("provider", config.llm_provider),
        ("model", config.model),
        ("temperature", config.llm_temperature),
    ])

    if config.category_overrides:
        lines.append("")
        lines.append("Per-Category Overrides:")
        for cat, overrides in config.category_overrides.items():
            lines.append(f"  {cat}: {overrides}")

    # Outlier + topic stats from results
    if results:
        for category, res in results.items():
            doc_counts = res.get("doc_counts", {})
            n_topics = len([t for t in doc_counts if t != -1])
            total = sum(doc_counts.values())
            outlier_count = doc_counts.get(-1, 0)
            pct = outlier_count / total * 100 if total else 0
            section(f"Results — {category}", [
                ("topics", n_topics),
                ("documents", total),
                ("outliers", f"{outlier_count} ({pct:.1f}%)"),
            ])

    lines.append("")
    lines.append("=" * 60 + "\n")

    with open(log_file, "w") as f:
        f.write("\n".join(lines))

    print(f"📝 Config logged to {log_file}")


def _drop_empty_topics(
    df: pd.DataFrame,
    labels: Dict[int, str],
    keywords_map: Dict[int, str],
    doc_counts: Dict[int, int],
    modeler: Optional["TopicModeler"] = None,
) -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
    """Reassign docs in keyword-empty topics to outliers (-1) and remove those topics.

    Also patches the BERTopic model's internal state so visualizations don't show them.
    """
    empty_ids = {
        tid for tid, kws in keywords_map.items()
        if tid != -1 and not any(k.strip() for k in kws.split(","))
    }
    if not empty_ids:
        return df, labels, keywords_map, doc_counts

    df = df.copy()
    df.loc[df["topic"].isin(empty_ids), "topic"] = -1
    for tid in empty_ids:
        labels.pop(tid, None)
        keywords_map.pop(tid, None)
        doc_counts.pop(tid, None)

    if modeler is not None:
        m = modeler.topic_model
        # Reassign docs in model's topic list
        m.topics_ = [-1 if t in empty_ids else t for t in m.topics_]
        # Update topic_sizes_
        for tid in empty_ids:
            n = m.topic_sizes_.pop(tid, 0)
            m.topic_sizes_[-1] = m.topic_sizes_.get(-1, 0) + n
        # Remove from representations — viz functions iterate over this dict
        for tid in empty_ids:
            m.topic_representations_.pop(tid, None)
            if m.topic_labels_ and tid in m.topic_labels_:
                del m.topic_labels_[tid]

    print(f"⚠️  Dropped {len(empty_ids)} empty topic(s): {sorted(empty_ids)}")
    return df, labels, keywords_map, doc_counts


def run_topic_modeling_pipeline(
    data_folder: str,
    output_folder: str = "./output",
    config: Optional[TopicModelConfig] = None,
) -> Dict[str, Any]:
    """
    Run complete topic modeling pipeline.

    Args:
        data_folder: Path to folder with CSV data files
        output_folder: Path to save outputs
        config: TopicModelConfig (uses defaults if None)

    Returns:
        Dictionary with results
    """
    start_time = time.time()
    config = config or TopicModelConfig()
    output_base = Path(output_folder)
    output_base.mkdir(parents=True, exist_ok=True)

    base = config.run_name
    existing = [int(d.name.split(f"{base}_")[1]) for d in output_base.glob(
        f"{base}_*") if d.is_dir() and d.name.split(f"{base}_")[1].isdigit()]
    run_dir = f"{base}_{max(existing, default=0) + 1:02d}"
    output_path = output_base / run_dir
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Run: {run_dir} → {output_path}")

    print("\n" + "="*60)
    print("📂 LOADING DATA")
    print("="*60)

    # Load both datasets
    datasets = {
        'barriers': load_csv_data(data_folder, 'barriers'),
        'motivators': load_csv_data(data_folder, 'motivators')
    }

    results = {}

    # Process each category with the same code
    for category, df in datasets.items():
        if df.empty:
            print(f"\n⚠️  No {category} data found, skipping...")
            continue

        print("\n" + "="*60)
        print(f"🎯 TOPIC MODELING: {category.upper()}")
        print("="*60)

        cat_config = config
        if config.category_overrides and category in config.category_overrides:
            cat_config = replace(config, **config.category_overrides[category])
            print(f"  📋 Overrides: {config.category_overrides[category]}")
        modeler = TopicModeler(cat_config)
        model_path = str(output_path / category)

        # Load cached embeddings if available (keyed by model name to avoid stale cache)
        model_slug = cat_config.embedding_model.split("/")[-1]
        embed_file = output_base / f"embeddings_{category}_{model_slug}.npy"
        embeddings = None
        if config.cache_embeddings and embed_file.exists():
            embeddings = np.load(str(embed_file))
            if len(embeddings) != len(df):
                print(f"⚠️ Embedding cache size mismatch, recomputing...")
                embeddings = None
            else:
                print(f"📂 Loaded cached embeddings from {embed_file}")

        # Always refit BERTopic (fast ~2s, only embeddings are slow ~10s)
        df, topics, probs = modeler.fit_transform(
            df, category, embeddings=embeddings)

        # Save embeddings after fit
        if config.cache_embeddings and modeler._embeddings is not None:
            np.save(str(embed_file), modeler._embeddings)
            print(f"💾 Cached embeddings to {embed_file}")

        # Save model for later inspection
        modeler.save(model_path)

        # Reduce outliers if enabled
        texts = df[category].tolist()
        if config.reduce_outliers:
            topics = modeler.reduce_outliers(
                texts, topics, strategy=config.reduce_outliers_strategy
            )
            df['topic'] = topics

        modeler.print_topics()

        # Generate LLM labels for topics
        labels, keywords_map, doc_counts = modeler.generate_topic_labels()
        df, labels, keywords_map, doc_counts = _drop_empty_topics(df, labels, keywords_map, doc_counts, modeler)
        modeler.set_topic_labels(labels)

        # Save labels to CSV (with keywords and doc counts for interpretability)
        labels_df = pd.DataFrame([
            {
                "topic_id": tid,
                "label": labels[tid],
                "doc_count": doc_counts[tid],
                "keywords": keywords_map[tid]
            }
            for tid in labels.keys()
        ])
        labels_df.to_csv(output_path / f"{category}_labels.csv", index=False)
        print(f"  ✓ Saved {category}_labels.csv")

        # Save topics CSV
        df.to_csv(output_path / f"{category}_topics.csv", index=False)

        results[category] = {
            'df': df,
            'topics': df['topic'].tolist(),
            'topic_info': modeler.get_topic_info(),
            'labels': labels,
            'keywords': keywords_map,
            'doc_counts': doc_counts
        }

        # Generate visualizations
        viz_results = modeler.viz(
            df=df,
            text_column=category,
            output_path=str(output_path),
            category=category
        )
        results[category]['viz'] = viz_results

        # Aggregate by year if available
        if 'year' in df.columns:
            yearly = aggregate_by_year(df)
            yearly.to_csv(output_path / f"{category}_yearly.csv")
            results[category]['yearly'] = yearly

        # Aggregate by company-year if both columns available
        if 'company' in df.columns and 'year' in df.columns:
            company_year = aggregate_by_company_year(df)
            company_year.to_csv(output_path / f"{category}_company_year.csv")
            results[category]['company_year'] = company_year
            print(f"  ✓ Saved {category}_company_year.csv")

        modeler.cleanup()

    duration_s = time.time() - start_time

    # Outlier summary
    outlier_parts = []
    for category, res in results.items():
        doc_counts = res.get("doc_counts", {})
        if -1 in doc_counts:
            outlier_count = doc_counts[-1]
            total = sum(doc_counts.values())
            pct = outlier_count / total * 100 if total else 0
            outlier_parts.append(
                f"{category} {outlier_count}/{total} ({pct:.1f}%)")

    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)
    print(f"📁 Results saved to: {output_path}")
    print(f"⏱️  Duration: {duration_s:.1f}s")
    if outlier_parts:
        print(f"📊 Outliers: {', '.join(outlier_parts)}")

    # Log config + results for reproducibility
    _write_config_log(output_path, config,
                      duration_s=duration_s, results=results)

    results["output_path"] = str(output_path)
    return results


def latest_run_dir(output_folder: str = "./output", run_name: str = "run") -> str:
    """Return path to the most recently created run directory."""
    runs = sorted(
        d for d in Path(output_folder).glob(f"{run_name}_*")
        if d.is_dir() and d.name.split(f"{run_name}_")[1].isdigit()
    )
    if not runs:
        raise ValueError(f"No run directories found in {output_folder}")
    return str(runs[-1])


def generate_deliverable_viz(
    run_dir: str,
    top_n: int = 6,
    config: Optional[TopicModelConfig] = None,
) -> None:
    """
    Generate filtered visualizations showing only the top N topics by doc count.

    Loads the merged model (saved by merge_topics_pipeline) for each category,
    picks the top N topics from labels.csv, and writes filtered HTML/PNG/CSV
    to {run_dir}/deliverable/.

    Falls back to the original model if no merged model exists (i.e. merge step
    was skipped).

    Args:
        run_dir: Path to run directory, e.g. "../out/topics/run_05"
        top_n: Number of top topics to include (by doc count)
        config: TopicModelConfig — only embedding_model matters here for loading
    """
    run_path = Path(run_dir)
    config = config or TopicModelConfig()
    deliverable_path = run_path / "deliverable"
    deliverable_path.mkdir(exist_ok=True)

    categories = [p.stem.replace("_labels", "") for p in sorted(run_path.glob("*_labels.csv"))]
    if not categories:
        print("⚠️  No labels CSV found — run merge_topics_pipeline first")
        return

    for category in categories:
        labels_df = pd.read_csv(run_path / f"{category}_labels.csv")
        top_ids = (
            labels_df[labels_df["topic_id"] != -1]
            .sort_values("doc_count", ascending=False)
            .head(top_n)["topic_id"]
            .tolist()
        )
        print(f"\n📊 {category}: top {top_n} topics → {top_ids}")

        # Save filtered labels and topics CSVs
        labels_df[labels_df["topic_id"].isin(top_ids)].to_csv(
            deliverable_path / f"{category}_labels.csv", index=False)
        df = pd.read_csv(run_path / f"{category}_topics.csv")
        df[df["topic"].isin(top_ids)].to_csv(
            deliverable_path / f"{category}_topics.csv", index=False)

        # Load merged model if available, otherwise original
        merged = str(run_path / f"{category}_merged")
        original = str(run_path / category)
        model_path = merged if os.path.exists(f"{merged}_model") else original

        modeler = TopicModeler(config)
        modeler.load(model_path)
        modeler.reduce_embeddings_for_viz()

        modeler.viz(
            df=df,
            text_column=category,
            output_path=str(deliverable_path),
            category=category,
            topics=top_ids,
        )
        modeler.cleanup()

    print(f"\n✓ Deliverable saved to {deliverable_path}")


def merge_topics_pipeline(
    run_dir: str,
    category: str,
    topics_to_merge: List[List[int]],
    config: Optional[TopicModelConfig] = None,
) -> pd.DataFrame:
    """
    Load a saved topic model, merge specified topic groups, re-label, and save in-place.

    Always loads from the original {category}_model saved by run_topic_modeling_pipeline,
    so re-running with different groups is safe — the original model is never overwritten.

    After merging, BERTopic keeps the lowest ID in each group (e.g. [7, 9, 12] → topic 7).
    Topic IDs that no longer exist are silently skipped, preventing IndexError on re-run.

    Args:
        run_dir: Path to run directory, e.g. "../out/topics/run_05"
        category: "barriers" or "motivators"
        topics_to_merge: List of topic ID groups to merge, e.g. [[7, 9, 12, 13], [2, 14]]
        config: TopicModelConfig for LLM label settings (uses defaults if None)

    Returns:
        Updated DataFrame with merged topic assignments
    """
    run_path = Path(run_dir)
    config = config or TopicModelConfig()
    model_path = str(run_path / category)

    modeler = TopicModeler(config)
    modeler.load(model_path)

    df = pd.read_csv(run_path / f"{category}_topics.csv")
    docs = df[category].tolist()

    # Validate topic IDs against the loaded model — skip IDs that no longer exist
    valid_ids = set(modeler.get_topic_info()["Topic"].tolist()) - {-1}
    filtered_merges = []
    for group in topics_to_merge:
        kept = [t for t in group if t in valid_ids]
        if len(kept) >= 2:
            filtered_merges.append(kept)
        elif kept:
            print(f"⚠️  Skipping {group}: only 1 ID still exists {kept}")
        else:
            print(f"⚠️  Skipping {group}: no IDs exist in current model")

    n_before = len(valid_ids)
    print(f"\n📊 Topics before merge: {n_before}")

    if not filtered_merges:
        print("⚠️  No valid merge groups — nothing to do")
        return df

    print(f"🔀 Merging {len(filtered_merges)} group(s):")
    for group in filtered_merges:
        print(f"  {group}")

    modeler.topic_model.merge_topics(docs, filtered_merges)

    df["topic"] = modeler.topic_model.topics_
    n_after = len(modeler.get_topic_info()[modeler.get_topic_info()["Topic"] != -1])
    print(f"✅ Topics after merge: {n_after}")

    labels, keywords_map, doc_counts = modeler.generate_topic_labels()
    df, labels, keywords_map, doc_counts = _drop_empty_topics(df, labels, keywords_map, doc_counts, modeler)
    modeler.set_topic_labels(labels)

    labels_df = pd.DataFrame([
        {"topic_id": tid, "label": labels[tid], "doc_count": doc_counts[tid], "keywords": keywords_map[tid]}
        for tid in labels.keys()
    ])
    labels_df.to_csv(run_path / f"{category}_labels.csv", index=False)
    df.to_csv(run_path / f"{category}_topics.csv", index=False)

    # Save merged model separately so generate_deliverable_viz can reload it
    modeler.save(str(run_path / f"{category}_merged"), save_embeddings=False)

    # Regenerate visualizations (embeddings loaded by load(), just need 2D reduction)
    modeler.reduce_embeddings_for_viz()
    modeler.viz(df=df, text_column=category, output_path=str(run_path), category=category)

    print(f"✓ Saved updated labels, topics CSV, and visualizations to {run_dir}")
    return df
