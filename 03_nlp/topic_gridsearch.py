"""Topic Grid Search — Staged Hyperparameter Tuning for BERTopic
=============================================================

Philosophy
----------
Good topic models require three layers of tuning:

  1. Embedding model — the foundation. Different models encode semantic
     relationships differently. Choose one that produces geometrically
     clean clusters for *this* corpus before tuning anything else.

  2. HDBSCAN structure — the main knob. Controls how many topics you get
     and how tight/loose the clusters are. Lock embedding first, then sweep.

  3. UMAP geometry — secondary but real. Controls how much structure
     HDBSCAN gets to work with. Lock HDBSCAN first, then sweep.

This module automates each stage and prints suggestions, but it is a
*suggestion tool*, not an oracle. Always inspect the saved CSVs and
override `gs.locked` before running the next stage if the auto-pick
looks wrong.

Usage
-----
    gs = TopicGridSearch("../out", "../out/topics", base_config)
    gs.stage1_embeddings(["BAAI/bge-small-en-v1.5", ...])
    # → inspect gs_stage1.csv, override gs.locked["embedding_model"] if needed
    gs.stage2_hdbscan()
    # → inspect gs_stage2_barriers.csv etc, override gs.locked["barriers"] if needed
    gs.stage3_umap()
    print(gs.category_overrides)  # paste into TopicModelConfig(category_overrides=...)

Notes on DBCV
-------------
DBCV (Density-Based Cluster Validity) measures how geometrically
well-separated and internally dense clusters are relative to the gaps
between them. Range is roughly -1 to +1 (higher = better). It is
computed separately from HDBSCAN's own objective, so it gives a second
opinion on cluster quality.

Limitations:
- DBCV knows nothing about topic *quality* or semantics. A high score
  means geometrically clean clusters, not necessarily interpretable ones.
  Always look at the actual topic words after running the pipeline.
- DBCV penalises outliers: outlier points contribute negatively to the
  score. A model that forces everything into clusters may score higher
  than one that correctly leaves genuine noise unclustered.

Scoring heuristic
-----------------
    score = dbcv * (1 - outlier_pct / 100)

This is a heuristic, not ground truth. The outlier penalty is a soft
assumption that high outlier rates usually indicate poorly-chosen params
(too-aggressive min_cluster_size) rather than genuine noise in the data.
This assumption is often wrong — some corpora have real noise that should
stay as outliers.

The filter thresholds (n_topics 8–40, outlier_pct < 30%) are heuristic
and calibrated loosely for this corpus size (~1200–1700 docs).

**Always check the CSV and override gs.locked if the suggestion looks wrong.**
"""

from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

import numpy as np
import pandas as pd

from .data_loader import load_csv_data
from .gpu_utils import clear_gpu_memory
from .topic_modelling import TopicModeler, TopicModelConfig, run_grid_search


CATEGORIES = ["barriers", "motivators"]


class TopicGridSearch:
    """
    Staged grid search for BERTopic hyperparameter tuning.

    Runs three stages:
      - Stage 1: Compare embedding models (encodes docs, runs quick HDBSCAN grid)
      - Stage 2: Sweep HDBSCAN structure (min_cluster_size, method, min_samples)
      - Stage 3: Sweep UMAP geometry (n_components, n_neighbors)

    State is stored in ``self.locked`` — a dict that accumulates best parameters
    found at each stage. Assign to it before running the next stage to override
    the auto-picked values.

    Example::

        gs = TopicGridSearch("../out", "../out/topics", base_config)
        gs.stage1_embeddings(["BAAI/bge-small-en-v1.5", "sentence-transformers/all-mpnet-base-v2"])
        gs.locked["embedding_model"] = "BAAI/bge-small-en-v1.5"  # override if needed
        gs.stage2_hdbscan()
        gs.stage3_umap()
        print(gs.category_overrides)
    """

    def __init__(self, state=None, data_folder: Optional[str] = None, output_folder: Optional[str] = None, base_config: Optional[TopicModelConfig] = None):
        """
        Args:
            state: PipelineState to extract data_folder, output_folder, base_config from (preferred)
            data_folder: Override data_folder (or pass via state)
            output_folder: Override output_folder (or pass via state)
            base_config: Base TopicModelConfig — UMAP, HDBSCAN, vectorizer params used as
                defaults unless overridden by locked params or param_grid being swept.
        """
        if state is not None:
            data_folder = data_folder or state.data_folder
            output_folder = output_folder or state.output_folder
            base_config = base_config or state.config
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.base_config = base_config or TopicModelConfig()
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        # Accumulated best params, filled in by each stage.
        # Keys:
        #   "embedding_model" (str)  — set by stage1, applies to all categories
        #   "barriers"        (dict) — HDBSCAN params from stage2 + UMAP from stage3
        #   "motivators"      (dict) — same
        self.locked: Dict[str, Any] = {}

    @property
    def category_overrides(self) -> dict:
        """
        Build the per-category overrides dict for TopicModelConfig(category_overrides=...).

        Returns a dict like::

            {
                "barriers":   {"embedding_model": "...", "hdbscan_min_cluster_size": 16, ...},
                "motivators": {"embedding_model": "...", "hdbscan_min_cluster_size": 25, ...},
            }

        Paste this into ``TopicModelConfig(category_overrides=...)`` in the main pipeline.
        """
        overrides = {}
        for cat in CATEGORIES:
            p = {}
            if "embedding_model" in self.locked:
                p["embedding_model"] = self.locked["embedding_model"]
            if cat in self.locked:
                p.update(self.locked[cat])
            if p:
                overrides[cat] = p
        return overrides

    def _print_candidates(self, df: pd.DataFrame, param_cols: List[str], category: str, n: int = 3):
        """
        Print top N candidates with per-row flags and a what-to-look-for guide.

        Ranks by ``dbcv * (1 - outlier_pct/100)`` after filtering to
        n_topics in [8, 40] and outlier_pct < 30%. Does NOT auto-lock anything —
        the user picks from the printed candidates or the saved CSV and sets
        gs.locked manually.

        Args:
            df: Grid search result DataFrame.
            param_cols: Column names to display per row (the swept params).
            category: "barriers" or "motivators" (for the header).
            n: Number of candidates to print.
        """
        df = df.copy()
        df["dbcv"] = pd.to_numeric(df["dbcv"], errors="coerce")
        valid = df[df["n_topics"].between(8, 40) & df["outlier_pct"].lt(30)].dropna(subset=["dbcv"])
        if valid.empty:
            valid = df.dropna(subset=["dbcv"])
        if valid.empty:
            print(f"  ⚠️  No valid rows found for {category}")
            return
        valid = valid.copy()
        valid["score"] = valid["dbcv"] * (1 - valid["outlier_pct"] / 100)
        top = valid.sort_values("score", ascending=False).head(n)

        print(f"\n📋 Top {n} candidates — {category} (see CSV for full results):")
        for i, (_, row) in enumerate(top.iterrows()):
            params = ", ".join(f"{c.split('_', 1)[-1]}={row[c]}" for c in param_cols if c in row.index)
            flags = []
            if row["n_topics"] < 10:
                flags.append("few topics")
            if row["n_topics"] > 35:
                flags.append("many topics")
            if row.get("largest_topic_pct", 0) > 40:
                flags.append("dominant topic")
            flag_str = "  ⚠️  " + ", ".join(flags) if flags else ""
            print(f"  #{i+1}  {params}  →  {int(row['n_topics'])} topics, "
                  f"{row['outlier_pct']}% outliers, DBCV={row['dbcv']:.4f}{flag_str}")

        print(f"\n  What to look for:")
        print(f"  - n_topics <10: over-merged — try smaller min_cluster_size or n_components")
        print(f"  - outlier_pct >25%: noisy — try larger n_neighbors or smaller min_cluster_size")
        print(f"  - largest_topic_pct >40%: one cluster dominates — params too coarse")
        print(f"  - DBCV close across candidates: prefer lower outliers and n_topics in 15–30")
        print(f"  - Compare across categories: similar UMAP geometry aids interpretability")

    def stage1_embeddings(self, models: List[Union[str, Tuple[str, int]]]) -> pd.DataFrame:
        """
        Stage 1: Compare embedding models using a quick HDBSCAN grid.

        For each model × category:
          - Encodes documents (~30s) and caches to
            ``embeddings_{category}_{model_slug}.npy``.
          - Runs a small representative HDBSCAN grid (3 combos) to get DBCV.
          - Reports max DBCV per model (more robust than a single run).

        The best model (highest avg best_score across categories) is locked in as
        ``gs.locked["embedding_model"]``. Override before running Stage 2 if needed.

        Saves: ``gs_stage1.csv`` (rows: model × category).

        Args:
            models: List of model entries. Each entry is either:
                - ``"org/model-name"`` — uses base_config.batch_size
                - ``("org/model-name", 16)`` — uses the given batch_size
                - ``("org/model-name", 16, "bfloat16")`` — also sets dtype

                batch_size only affects encode speed, not embedding values,
                so mixing sizes across models does not affect comparability.
                dtype ("bfloat16"/"float16") halves VRAM for LLM-based embedders
                (e.g. Qwen3) that SentenceTransformer otherwise loads in float32.

        Returns:
            DataFrame with columns: model, batch_size, category, max_dbcv,
            best_n_topics, best_outlier_pct, best_score.
        """
        # Small representative grid — enough for a robust DBCV estimate
        # without spending too long. 3 combos × n_models × 2 categories.
        quick_grid = {
            "hdbscan_min_cluster_size": [12, 16, 20],
            "hdbscan_cluster_selection_method": ["eom"],
            "hdbscan_min_samples": [3],
        }

        rows = []
        failed: Dict[str, str] = {}  # model_name -> error message

        for entry in models:
            if isinstance(entry, tuple) and len(entry) == 3:
                model_name, batch_size, embedding_dtype = entry
            elif isinstance(entry, tuple):
                model_name, batch_size = entry
                embedding_dtype = self.base_config.embedding_dtype  # inherit default (e.g. "bfloat16")
            else:
                model_name = entry
                batch_size = self.base_config.batch_size
                embedding_dtype = self.base_config.embedding_dtype  # inherit default
            model_slug = model_name.split("/")[-1]
            print(f"\n{'='*60}")
            print(f"📐 Stage 1 — Embedding: {model_name}")
            print(f"{'='*60}")

            # Encode and cache embeddings per category
            encode_failed = False
            for cat in CATEGORIES:
                embed_file = Path(self.output_folder) / \
                    f"embeddings_{cat}_{model_slug}.npy"
                if embed_file.exists():
                    print(
                        f"  ✓ {cat}: using cached embeddings ({embed_file.name})")
                    continue

                df = load_csv_data(self.data_folder, cat)
                if df.empty:
                    print(f"  ⚠️ No {cat} data found, skipping")
                    continue

                print(
                    f"  🧮 {cat}: encoding {len(df)} docs (batch_size={batch_size})...")

                kwargs = dict(
                    embedding_model=model_name,
                    batch_size=batch_size,
                    verbose=False
                )
                if embedding_dtype is not None:
                    kwargs["embedding_dtype"] = embedding_dtype

                model_config = replace(self.base_config, **kwargs)

                modeler = TopicModeler(model_config)
                try:
                    embeddings = modeler.encode_documents(df[cat].tolist())
                    np.save(str(embed_file), embeddings)
                    print(f"  💾 Saved {embed_file.name} {embeddings.shape}")
                except Exception as e:
                    err = str(e)
                    kind = "OOM" if "out of memory" in err.lower() else "ERROR"
                    print(f"  ⚠️ {kind} encoding {cat}: {err[:120]}")
                    failed[model_name] = f"{kind}: {err[:200]}"
                    encode_failed = True
                finally:
                    modeler.cleanup()
                    clear_gpu_memory()

                if encode_failed:
                    break  # don't attempt the other category with a failing model

            if encode_failed:
                for cat in CATEGORIES:
                    rows.append({"model": model_name, "batch_size": batch_size, "category": cat,
                                 "max_dbcv": None, "best_n_topics": None, "best_outlier_pct": None,
                                 "best_score": None, "error": failed[model_name]})
                continue

            # Quick HDBSCAN grid per category to measure cluster quality
            for cat in CATEGORIES:
                embed_file = Path(self.output_folder) / \
                    f"embeddings_{cat}_{model_slug}.npy"
                if not embed_file.exists():
                    continue

                kwargs = dict(
                    embedding_model=model_name,
                    batch_size=batch_size,
                    verbose=False
                )
                if embedding_dtype is not None:
                    kwargs["embedding_dtype"] = embedding_dtype

                model_config = replace(self.base_config, **kwargs)
                try:
                    gs_df = run_grid_search(
                        self.data_folder,
                        self.output_folder,
                        category=cat,
                        config=model_config,
                        param_grid=quick_grid,
                    )
                    # Score each combo: dbcv * (1 - outlier_pct/100), filtered to sane range
                    _s = gs_df.copy()
                    _s["dbcv"] = pd.to_numeric(_s["dbcv"], errors="coerce")
                    _valid = _s[_s["n_topics"].between(8, 40) & _s["outlier_pct"].lt(30)].dropna(subset=["dbcv"])
                    if _valid.empty:
                        _valid = _s.dropna(subset=["dbcv"])
                    _valid = _valid.copy()
                    _valid["score"] = _valid["dbcv"] * (1 - _valid["outlier_pct"] / 100)
                    best = _valid.sort_values("score", ascending=False).iloc[0] if not _valid.empty else _s.iloc[0]
                    rows.append({
                        "model": model_name,
                        "batch_size": batch_size,
                        "category": cat,
                        "max_dbcv": round(float(gs_df["dbcv"].dropna().max()), 4) if not gs_df["dbcv"].dropna().empty else None,
                        "best_n_topics": int(best["n_topics"]),
                        "best_outlier_pct": float(best["outlier_pct"]),
                        "best_score": round(float(best["score"]), 4) if "score" in best.index else None,
                        "error": None,
                    })
                except Exception as e:
                    err = str(e)
                    print(f"  ⚠️ Grid search failed for {cat}: {err[:120]}")
                    rows.append({"model": model_name, "batch_size": batch_size, "category": cat,
                                 "max_dbcv": None, "best_n_topics": None, "best_outlier_pct": None,
                                 "best_score": None, "error": f"GRID: {err[:200]}"})

        result_df = pd.DataFrame(rows)

        # Pick best model: highest avg best_score across categories (exclude failed).
        # best_score = dbcv * (1 - outlier_pct/100) from the best *valid* combo per category,
        # which respects the n_topics filter. max_dbcv is NOT used here because it can be
        # inflated by combos that were filtered out (e.g. n_topics > 40).
        best_model = self.base_config.embedding_model
        successful = result_df[result_df["error"].isna(
        )] if "error" in result_df.columns else result_df
        if not successful.empty and successful["best_score"].notna().any():
            avg_score = successful.dropna(subset=["best_score"]).groupby("model")[
                "best_score"].mean()
            best_model = avg_score.idxmax()
            self.locked["embedding_model"] = best_model

        print(f"\n{'='*60}")
        print("📊 Stage 1 Results")
        print(f"{'='*60}")
        if not result_df.empty:
            print(result_df.to_string(index=False))

        if failed:
            print(f"\n⚠️  Failed models ({len(failed)}/{len(models)}):")
            for m, err in failed.items():
                print(f"   {m}: {err[:100]}")
            if len(failed) == len(models):
                raise RuntimeError(
                    "All models failed during Stage 1 — check GPU memory, batch sizes, or model availability."
                )

        if best_model != self.base_config.embedding_model:
            print(f"\n✅ Suggestion: embedding_model = \"{best_model}\"")
        print(f"   Override: gs.locked[\"embedding_model\"] = \"<model>\"")

        out_path = Path(self.output_folder) / "gs_stage1.csv"
        result_df.to_csv(out_path, index=False)
        print(f"\n💾 Saved {out_path}")

        return result_df

    def stage2_hdbscan(
        self,
        param_grid: Optional[Dict[str, List]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Stage 2: Sweep HDBSCAN structure parameters per category.

        Uses ``gs.locked["embedding_model"]`` (set by stage1 or manually).

        Default grid (42 combos)::

            hdbscan_min_cluster_size:        [8, 12, 16, 20, 25, 30, 40]
            hdbscan_cluster_selection_method: ["eom", "leaf"]
            hdbscan_min_samples:             [2, 3, 5]

        Auto-picks best per category via the score heuristic (see module
        docstring), filtered to n_topics in [8, 40] and outlier_pct < 30%.
        Sets ``gs.locked["barriers"]`` and ``gs.locked["motivators"]``.

        Saves: ``gs_stage2_barriers.csv``, ``gs_stage2_motivators.csv``.

        Args:
            param_grid: Override the default HDBSCAN parameter grid.

        Returns:
            Dict mapping category name → result DataFrame.
        """
        if "embedding_model" not in self.locked:
            print(
                "⚠️  No embedding_model in gs.locked — using base_config.embedding_model")
            print(
                "    Run stage1_embeddings() first, or set gs.locked[\"embedding_model\"] manually.")

        embedding_model = self.locked.get(
            "embedding_model", self.base_config.embedding_model)

        if param_grid is None:
            param_grid = {
                "hdbscan_min_cluster_size": [8, 12, 16, 20, 25, 30, 40],
                "hdbscan_cluster_selection_method": ["eom", "leaf"],
                "hdbscan_min_samples": [2, 3, 5],
            }

        results = {}
        for cat in CATEGORIES:
            print(f"\n{'='*60}")
            print(f"🔍 Stage 2 — HDBSCAN — {cat}")
            print(f"{'='*60}")

            cat_config = replace(
                self.base_config, embedding_model=embedding_model, verbose=False)
            gs_df = run_grid_search(
                self.data_folder,
                self.output_folder,
                category=cat,
                config=cat_config,
                param_grid=param_grid,
            )

            out_path = Path(self.output_folder) / f"gs_stage2_{cat}.csv"
            gs_df.to_csv(out_path, index=False)
            print(f"💾 Saved {out_path}")

            self._print_candidates(
                gs_df,
                ["hdbscan_min_cluster_size", "hdbscan_cluster_selection_method", "hdbscan_min_samples"],
                cat,
            )
            print(f"\n  Set params: gs.locked[\"{cat}\"] = {{\"hdbscan_min_cluster_size\": ..., ...}}")

            results[cat] = gs_df

        print(f"\n📋 Set gs.locked per category before running Stage 3, e.g.:")
        print(f"   gs.locked[\"barriers\"] = {{\"hdbscan_min_cluster_size\": 25, \"hdbscan_cluster_selection_method\": \"eom\", \"hdbscan_min_samples\": 5}}")
        print(f"   gs.locked[\"motivators\"] = {{...}}")

        return results

    def stage3_umap(
        self,
        param_grid: Optional[Dict[str, List]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Stage 3: Sweep UMAP geometry parameters per category.

        Uses ``gs.locked["embedding_model"]`` and per-category HDBSCAN params
        from stage2 (or base_config defaults if stage2 wasn't run).

        Default grid (20 combos)::

            umap_n_components: [5, 10, 15, 30, 50]
            umap_n_neighbors:  [10, 15, 25, 40]

        Best UMAP params are merged into ``gs.locked["barriers"]`` and
        ``gs.locked["motivators"]`` alongside the stage2 HDBSCAN params.

        Saves: ``gs_stage3_barriers.csv``, ``gs_stage3_motivators.csv``.

        Args:
            param_grid: Override the default UMAP parameter grid.

        Returns:
            Dict mapping category name → result DataFrame.
        """
        if "embedding_model" not in self.locked:
            print(
                "⚠️  No embedding_model in gs.locked — using base_config.embedding_model")
        if not any(cat in self.locked for cat in CATEGORIES):
            print("⚠️  No HDBSCAN params in gs.locked — run stage2_hdbscan() first")

        embedding_model = self.locked.get(
            "embedding_model", self.base_config.embedding_model)

        if param_grid is None:
            param_grid = {
                "umap_n_components": [5, 10, 15, 30, 50],
                "umap_n_neighbors": [10, 15, 25, 40],
            }

        results = {}
        for cat in CATEGORIES:
            print(f"\n{'='*60}")
            print(f"🗺️  Stage 3 — UMAP — {cat}")
            print(f"{'='*60}")

            # Apply locked embedding + per-cat HDBSCAN overrides from stage2
            cat_config = replace(
                self.base_config, embedding_model=embedding_model, verbose=False)
            if cat in self.locked:
                cat_config = replace(cat_config, **self.locked[cat])

            gs_df = run_grid_search(
                self.data_folder,
                self.output_folder,
                category=cat,
                config=cat_config,
                param_grid=param_grid,
            )

            out_path = Path(self.output_folder) / f"gs_stage3_{cat}.csv"
            gs_df.to_csv(out_path, index=False)
            print(f"💾 Saved {out_path}")

            self._print_candidates(gs_df, ["umap_n_components", "umap_n_neighbors"], cat)
            print(f"\n  Set params: gs.locked[\"{cat}\"].update({{\"umap_n_components\": ..., \"umap_n_neighbors\": ...}})")

            results[cat] = gs_df

        print(f"\n📋 After setting gs.locked, paste into TopicModelConfig:")
        print(f"   print(gs.category_overrides)")
        if self.category_overrides:
            print(self.category_overrides)

        return results
