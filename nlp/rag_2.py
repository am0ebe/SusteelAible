"""
RAG Topic Modeling Pipeline
============================

Topic modeling, categorization and visualization using BERTopic.

Key components:
- Embedding: thenlper/gte-small (or sentence-transformers/all-mpnet-base-v2)
- Dimensionality Reduction: UMAP
- Clustering: HDBSCAN (dynamic cluster sizes + outlier detection)
- Topic Representation: KeyBERTInspired for better topic words
- Visualization: DataMapPlot for publication-ready figures

Usage:
    python rag_2.py --data-folder ./data --output-folder ./output

Requirements:
    pip install bertopic sentence-transformers umap-learn hdbscan
    pip install datamapplot  # Optional: for DataMapPlot visualization
"""

from nlp import load_csv_data
from nlp import GPUManager
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, asdict

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


# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Local imports


# ==================== Configuration ====================

@dataclass
class TopicModelConfig:
    """Configuration for topic modeling pipeline."""

    # Embedding model
    embedding_model: str = "snowflake-arctic-embed-s"  # top MTEB for size
    # embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    # embedding_model: str = "Octen/Octen-Embedding-4B"
    # embedding_model: str = "bge-small-en-v1.5" # ⚡/384d

    batch_size: int = 64  # embed (increase if GPU memory allows)

    # UMAP parameters
    umap_n_neighbors: int = 30
    umap_n_components: int = 15  # Reduce to 5D for clustering
    umap_min_dist: float = 0.05  # higher = worse cluster, but better viz
    umap_metric: str = 'cosine'  # 'cosine'
    umap_random_state: int = 42

    # HDBSCAN parameters
    # min_cluster_size controls number of topics (higher = fewer topics)
    hdbscan_min_cluster_size: int = 15  # 1-3% datasize > 0.02 * 1000 cluster // 15
    hdbscan_min_samples: int = 2  # Lower = less noise/outliers
    hdbscan_metric: str = 'euclidean'
    hdbscan_cluster_selection_method: str = 'eom'  # 'eom' or 'leaf'

    vectorizer_ngram_range: Tuple[int, int] = (1, 2)  # Include bigrams
    # Minimum document frequency (use 1 for small topics, 2+ for large datasets)
    vectorizer_min_df: int = 1
    vectorizer_max_df: float = 0.95  # rm common >95%

    mmr_diversity: float = 0.3

    # BERTopic parameters
    top_n_words: int = 10
    # Set to reduce topics post-hoc # can use auto?
    nr_topics: Optional[int] = 7
    # Set True for soft clustering (slower)
    calculate_probabilities: bool = False
    # Reduce outliers by assigning to nearest topic (post-hoc)
    reduce_outliers: bool = True
    # 'embeddings', 'c-tf-idf', or 'distributions'
    reduce_outliers_strategy: str = "embeddings"

    # Processing
    verbose: bool = True

    # Visualization
    # 2D UMAP for visualization (separate from clustering)
    viz_umap_n_neighbors: int = 10  # +
    viz_umap_n_components: int = 2
    viz_umap_min_dist: float = 0.0

    # LLM settings (Ollama) for topic labeling
    ollama_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.0

    # Embedding cache (save immediately after computation to survive crashes)
    embeddings_cache_path: Optional[str] = None  # e.g., "out/embeddings_cache"


# ==================== Prompt ====================

TOPIC_LABEL_PROMPT = ChatPromptTemplate.from_template("""You are an expert analyst specializing in thematic classification and topic labeling.

KEYWORDS:
{keywords}

IMPORTANT:
If the output violates ANY rule below, it is incorrect.
You must self-check before answering.

OUTPUT RULES:
Return ONLY the label text
No punctuation
No quotes
No comments
No extra text

INSTRUCTIONS:
Read all keywords carefully
Identify the single common underlying issue they represent
Generate a concise, stakeholder-ready label

LABEL REQUIREMENTS:
Length: 3–5 words
Style: Title Case
Use & instead of "and" where appropriate
Describe a specific barrier or motivator
Avoid generic or abstract terms

A GOOD LABEL:
Names a concrete issue
Uses a clear noun phrase
Describes the issue itself, not the documents

DO NOT:
Explain reasoning
Mention keywords or documents
Use meta terms like Theme, Topic, or Issue

EXAMPLES OF GOOD LABELS:
Raw Materials & Energy Availability
Fossil-Free Steel Innovation
Import Competition & Overcapacity
Environmental Permits & Compliance

EXAMPLES OF BAD LABELS:
Operations
Identify Barriers
Cost
SSAB Production Issues
Various challenges in the steel production process

Before answering, internally verify all rules are satisfied.
Then output the label only.
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
    "million", "billion", "EUR", "USD", "barriers", "barrier", "risk", "risks"
    "mention", "mentioned", "mentions",
    "qualifying", "motivators", "motivating", "motivator",
    # Filler words that sometimes appear
    "also", "including", "various", "related", "based", "using",
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
        """Lazy-load the Ollama LLM."""
        if self._llm is None:
            self._log(f"Loading Ollama model: {self.config.ollama_model}")
            self._llm = ChatOllama(
                model=self.config.ollama_model,
                base_url=self.config.ollama_base_url,
                temperature=self.config.llm_temperature,
            )
        return self._llm

    def _load_embedding_model(self):
        """Load sentence transformer embedding model."""

        self._log(
            f"\n🤖 Loading embedding model: {self.config.embedding_model}")
        self._embedding_model = SentenceTransformer(
            self.config.embedding_model)

        # Move to GPU if available
        if self.gpu.is_cuda:
            self._embedding_model = self._embedding_model.to(self.gpu.device)
            self._log("✅ Using GPU for encoding")
        else:
            self._log("⚠️  Using CPU for encoding (slower)")

    def _create_topic_model(self):
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
            embedding_model=self.embedding_model,
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

        # Generate or load embeddings
        cache_path = self.config.embeddings_cache_path
        if embeddings is None and cache_path:
            cache_file = f"{cache_path}_{text_column}.npy"
            if os.path.exists(cache_file):
                self._log(f"📂 Loading cached embeddings from {cache_file}")
                embeddings = np.load(cache_file)
                if len(embeddings) != len(texts):
                    self._log(
                        f"⚠️ Cache size mismatch ({len(embeddings)} vs {len(texts)}), recomputing...")
                    embeddings = None

        if embeddings is None:
            embeddings = self.encode_documents(texts)
            # Save immediately to survive crashes
            if cache_path:
                cache_file = f"{cache_path}_{text_column}.npy"
                os.makedirs(os.path.dirname(cache_file), exist_ok=True) if os.path.dirname(
                    cache_file) else None
                np.save(cache_file, embeddings)
                self._log(f"💾 Cached embeddings to {cache_file}")

        self._embeddings = embeddings

        # Create and fit topic model
        self._topic_model = self._create_topic_model()

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
        Generate human-readable labels for topics using Ollama LLM.

        Uses the topic keywords from get_topic_info() to generate
        short descriptive labels (2-4 words) for each topic.

        Returns:
            Tuple of (labels_dict, keywords_dict, doc_count_dict)
        """
        topic_info = self.get_topic_info()
        labels = {}
        keywords_map = {}
        doc_counts = {}

        self._log("\n🏷️  Generating topic labels with LLM...")

        chain = TOPIC_LABEL_PROMPT | self.llm

        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            doc_counts[topic_id] = row['Count']

            # Handle outlier topic - still get keywords for analysis
            if topic_id == -1:
                labels[topic_id] = "Outliers (unassigned)"
                top_words = self.topic_model.get_topic(topic_id)
                if top_words:
                    keywords_map[topic_id] = ", ".join(
                        [w for w, _ in top_words[:10]])
                else:
                    keywords_map[topic_id] = ""
                continue

            # Get keywords for this topic
            top_words = self.topic_model.get_topic(topic_id)
            if not top_words:
                labels[topic_id] = f"Topic {topic_id}"
                keywords_map[topic_id] = ""
                continue

            keywords_raw = ", ".join([word for word, _ in top_words[:10]])
            keywords_filtered = _filter_keywords(keywords_raw)
            # Store original for reference
            keywords_map[topic_id] = keywords_raw

            try:
                response = chain.invoke({"keywords": keywords_filtered})
                label = response.content if hasattr(
                    response, "content") else str(response)
                label = label.strip().strip('"\'')
                labels[topic_id] = label
                self._log(f"  Topic {topic_id}: {label}")
            except Exception as e:
                self._log(f"  ⚠️ Topic {topic_id}: Error - {e}")
                labels[topic_id] = f"Topic {topic_id}"

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
        top_n_topics: int = 10
    ) -> Dict[str, Any]:
        """
        Generate and save all visualizations for a category.

        Args:
            df: DataFrame with text data and metadata (year, company columns)
            text_column: Name of column containing document text
            output_path: Directory to save visualizations
            category: Category name ("barriers" or "motivators")
            top_n_topics: Number of topics for barchart

        Returns:
            Dict with paths to saved files and any errors
        """
        output = Path(output_path)
        output.mkdir(parents=True, exist_ok=True)

        docs = df[text_column].tolist()
        self._log(f"\n📊 Generating visualizations for {category}...")

        results = {"saved": [], "errors": []}

        # Ensure 2D embeddings are computed for document visualizations
        if self._reduced_embeddings is None and self._embeddings is not None:
            self.reduce_embeddings_for_viz()

        viz_configs = [
            ("barchart", lambda: self.topic_model.visualize_barchart(
                top_n_topics=top_n_topics)),
            ("topics_2d", lambda: self.topic_model.visualize_topics()),
            ("hierarchy", lambda: self.topic_model.visualize_hierarchy()),
            ("heatmap", lambda: self.topic_model.visualize_heatmap()),
            ("documents", lambda: self.topic_model.visualize_documents(
                docs,
                embeddings=self._embeddings,
                reduced_embeddings=self._reduced_embeddings
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
                years = df['year'].tolist()
                topics_over_time = self.topic_model.topics_over_time(
                    docs, years,
                    global_tuning=True,
                    evolution_tuning=True
                )
                fig = self.topic_model.visualize_topics_over_time(
                    topics_over_time,
                    top_n_topics=top_n_topics
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
                companies = df['company'].tolist()
                topics_per_company = self.topic_model.topics_per_class(
                    docs, companies)
                fig = self.topic_model.visualize_topics_per_class(
                    topics_per_company,
                    top_n_topics=top_n_topics
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
                embeddings=self._embeddings,
                reduced_embeddings=self._reduced_embeddings,
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


# ==================== Main Pipeline ====================

def _write_config_log(output_path: Path, config: TopicModelConfig, force_retrain: bool):
    """Write config parameters to a log file for reproducibility."""
    log_file = output_path / "config_log.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"{'='*60}",
        f"Topic Modeling Pipeline Run",
        f"Timestamp: {timestamp}",
        f"Force Retrain: {force_retrain}",
        f"{'='*60}",
        "",
        "TopicModelConfig:",
        "-" * 40,
    ]

    for key, value in asdict(config).items():
        lines.append(f"  {key}: {value}")

    lines.append("")
    lines.append("=" * 60 + "\n")

    # write to log file
    with open(log_file, "w") as f:
        f.write("\n".join(lines))

    print(f"📝 Config logged to {log_file}")


def run_topic_modeling_pipeline(
    data_folder: str,
    output_folder: str = "./output",
    config: Optional[TopicModelConfig] = None,
    force_retrain: bool = False
) -> Dict[str, Any]:
    """
    Run complete topic modeling pipeline.

    Args:
        data_folder: Path to folder with CSV data files
        output_folder: Path to save outputs
        config: TopicModelConfig (uses defaults if None)
        force_retrain: If True, ignore cached model/embeddings and retrain from scratch

    Returns:
        Dictionary with results
    """
    config = config or TopicModelConfig()
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Log config for reproducibility
    _write_config_log(output_path, config, force_retrain)

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

        modeler = TopicModeler(config)
        model_path = str(output_path / category)

        # Try to load cached model + embeddings (unless force_retrain)
        if not force_retrain and os.path.exists(f"{model_path}_model"):
            print(f"📂 Loading cached model from {model_path}_model")
            modeler.load(model_path)
            texts = df[category].tolist()
            topics, probs = modeler.topic_model.transform(
                texts, modeler._embeddings)
            df = df.copy()
            df['topic'] = topics
        else:
            if force_retrain:
                print("🔄 Force retrain enabled, ignoring cache")
            # Fit topic model from scratch
            df, topics, probs = modeler.fit_transform(df, category)
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
            'topics': topics,
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

    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)
    print(f"📁 Results saved to: {output_path}")

    return results
