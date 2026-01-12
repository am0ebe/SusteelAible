"""
ClimateBERT Visualization & Analysis Module
============================================

Generates plots, CSVs, and analysis from BERT pipeline cache files.

Usage:
    from bert_2 import ClimateBERTVisualizer, visualize_results

    # Quick start (after running bert_pipeline)
    visualize_results("cache", "")

    # Detailed usage
    viz = ClimateBERTVisualizer(cache_dir="cache", output_dir="out")
    viz.load_data()
    viz.export_csvs()
    viz.generate_all_plots()
    viz.print_summary()
"""

import json
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class VisualizationConfig:
    """Configuration for visualization module."""

    cache_dir: str = "cache"
    output_dir: str = "out"

    # Exclude incomplete years (since we don't have all data for the current/last year,
    # including it skews the plots and makes trends look artificially different)
    exclude_year_gte: int = 2025

    # Plot style
    figsize_large: Tuple[int, int] = (12, 7)
    figsize_medium: Tuple[int, int] = (10, 6)
    dpi: int = 200


# =============================================================================
# WORD FREQUENCY UTILITIES
# =============================================================================

# Stopwords to exclude
STOPWORDS = {
    # English stopwords
    "the", "and", "for", "that", "with", "are", "from", "this", "was", "were",
    "been", "have", "has", "will", "would", "could", "should", "their", "they",
    "which", "what", "when", "where", "who", "how", "our", "your", "more", "also",
    "than", "into", "over", "such", "through", "being", "between", "after", "before",
    "these", "those", "other", "some", "most", "about", "including", "within",
    # Domain words to exclude
    "steel", "company", "group", "year", "years", "report", "reporting", "annual",
    # Company names
    "arcelormittal", "arcelor", "mittal", "thyssenkrupp", "thyssen", "krupp",
    "voestalpine", "salzgitter", "ssab", "outokumpu", "tata", "tatasteel",
    "nippon", "nipponsteel", "baosteel", "baoshan", "posco", "nucor",
    "acerinox", "celsa", "dillinger", "sidenor", "feralpi", "arvedi",
}

# Word groups: map variants to single representative
WORD_GROUPS = {
    "emission": ["emission", "emissions", "emitting", "emitted", "emit"],
    "carbon": ["carbon", "carbons", "co2", "dioxide"],
    "energy": ["energy", "energies", "energetic"],
    "sustainable": ["sustainable", "sustainability", "sustainably"],
    "environment": ["environment", "environmental", "environmentally"],
    "climate": ["climate", "climatic", "climates"],
    "reduction": ["reduction", "reductions", "reduce", "reduced", "reducing"],
    "target": ["target", "targets", "targeted", "targeting"],
    "goal": ["goal", "goals"],
    "commitment": ["commitment", "commitments", "commit", "committed"],
    "production": ["production", "produce", "produced", "producing"],
    "technology": ["technology", "technologies", "technological"],
    "innovation": ["innovation", "innovations", "innovative"],
    "investment": ["investment", "investments", "invest", "invested"],
    "development": ["development", "develop", "developed", "developing"],
    "strategy": ["strategy", "strategies", "strategic"],
    "efficiency": ["efficiency", "efficient", "efficiently"],
    "renewable": ["renewable", "renewables"],
    "hydrogen": ["hydrogen"],
    "recycling": ["recycling", "recycle", "recycled", "recyclable"],
    "circular": ["circular", "circularity"],
    "decarbonization": ["decarbonization", "decarbonisation", "decarbonize"],
    "net-zero": ["netzero", "zero", "neutrality", "neutral"],
}


def get_word_frequencies(texts: List[str], min_word_len: int = 4, top_n: int = 50) -> List[Tuple[str, int]]:
    """Get word frequencies with grouping."""
    # Build reverse lookup
    variant_to_rep = {}
    for rep, variants in WORD_GROUPS.items():
        for v in variants:
            variant_to_rep[v] = rep

    all_words = []
    for text in texts:
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        for w in words:
            if len(w) >= min_word_len and w not in STOPWORDS:
                all_words.append(variant_to_rep.get(w, w))

    return Counter(all_words).most_common(top_n)


# =============================================================================
# MAIN VISUALIZER CLASS
# =============================================================================

class ClimateBERTVisualizer:
    """
    Generates visualizations and exports from BERT pipeline results.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()

        self.cache_dir = Path(self.config.cache_dir)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data frames
        self.report_df: Optional[pd.DataFrame] = None
        self.cy_df: Optional[pd.DataFrame] = None  # Company-Year aggregated

        # Set plot style
        plt.style.use("seaborn-v0_8-whitegrid")

    def _save_plot(self, name: str):
        """Save current plot."""
        plt.tight_layout()
        filepath = self.output_dir / name
        plt.savefig(filepath, dpi=self.config.dpi, bbox_inches="tight")
        print(f"   ✓ {name}")
        plt.close()

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and process BERT cache files."""
        bert_files = list(self.cache_dir.glob("*_bert.json"))

        if not bert_files:
            raise FileNotFoundError(
                f"No BERT cache files found in {self.cache_dir}")

        rows = []
        for fp in bert_files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)

                chunks = data.get("chunks", [])
                if not chunks:
                    continue

                n = len(chunks)
                detector = np.array([c.get("detector_score", 0)
                                    for c in chunks])
                weights = detector if detector.sum() > 0 else np.ones(n)

                # Extract labels
                spec_labels = [c.get("specificity_label", "non")
                               for c in chunks]
                sent_labels = [c.get("sentiment_label", "neutral")
                               for c in chunks]
                comm_labels = [c.get("commitment_label", "no") for c in chunks]
                netzero_labels = [c.get("netzero_label", "none")
                                  for c in chunks]

                # Sentiment weights
                opp_w = sum(w for w, l in zip(weights, sent_labels)
                            if l == "opportunity")
                risk_w = sum(w for w, l in zip(
                    weights, sent_labels) if l == "risk")
                neutral_w = sum(w for w, l in zip(
                    weights, sent_labels) if l == "neutral")
                total_w = weights.sum()

                # Netzero counts
                netzero_count = sum(
                    1 for l in netzero_labels if l == "reduction")
                netzero_pct = (netzero_count / n * 100) if n > 0 else 0

                # N0 subset metrics
                n0_chunks = [c for c in chunks if c.get(
                    "netzero_label") == "reduction"]
                non_n0_chunks = [c for c in chunks if c.get(
                    "netzero_label") != "reduction"]

                def safe_mean(lst, key, default=0):
                    vals = [c.get(key, default) for c in lst]
                    return np.mean(vals) if vals else default

                rows.append({
                    "company": data.get("company", "Unknown"),
                    "company_id": str(data.get("company_id", "unknown")),
                    "year": int(data.get("year", 0)),
                    "is_translated": data.get("translated", False),
                    # Funnel metrics
                    "total_chunks": data.get("total_chunks", 0),
                    "climate_chunks": data.get("climate_chunks", n),
                    "climate_pct": data.get("kept_percentage", 100.0),
                    "netzero_chunks": netzero_count,
                    "netzero_pct": netzero_pct,
                    # Quality scores (weighted)
                    "specificity_w": np.average(
                        [c.get("specificity_score", 0) for c in chunks], weights=weights
                    ),
                    "commitment_w": np.average(
                        [c.get("commitment_score", 0) for c in chunks], weights=weights
                    ),
                    # N0 subset metrics
                    "n0_specificity": safe_mean(n0_chunks, "specificity_score"),
                    "n0_commitment": safe_mean(n0_chunks, "commitment_score"),
                    "non_n0_specificity": safe_mean(non_n0_chunks, "specificity_score"),
                    "non_n0_commitment": safe_mean(non_n0_chunks, "commitment_score"),
                    # Sentiment percentages
                    "opportunity_pct": (opp_w / total_w * 100) if total_w > 0 else 0,
                    "risk_pct": (risk_w / total_w * 100) if total_w > 0 else 0,
                    "neutral_pct": (neutral_w / total_w * 100) if total_w > 0 else 0,
                })
            except Exception as e:
                print(f"⚠️ Error loading {fp.name}: {e}")
                continue

        df = pd.DataFrame(rows)

        # Filter by year
        if self.config.exclude_year_gte:
            df = df[df["year"] < self.config.exclude_year_gte]

        if df.empty:
            raise ValueError("No valid data after filtering")

        # Aggregate to company-year
        cy = df.groupby(["company_id", "year"]).agg({
            "company": "first",
            "is_translated": "any",
            "total_chunks": "sum",
            "climate_chunks": "sum",
            "climate_pct": "mean",
            "netzero_chunks": "sum",
            "netzero_pct": "mean",
            "specificity_w": "mean",
            "commitment_w": "mean",
            "n0_specificity": "mean",
            "n0_commitment": "mean",
            "non_n0_specificity": "mean",
            "non_n0_commitment": "mean",
            "opportunity_pct": "mean",
            "risk_pct": "mean",
            "neutral_pct": "mean",
        }).reset_index()

        # Derived metrics
        cy["talk_score"] = (
            (cy["climate_pct"] / 100) * 0.20 +
            cy["specificity_w"] * 0.40 +
            cy["commitment_w"] * 0.40
        ) * 100

        cy["n0_talk_score"] = (
            (cy["netzero_pct"] / 100) * 0.30 +
            cy["n0_specificity"] * 0.35 +
            cy["n0_commitment"] * 0.35
        ) * 100

        # Sentiment balances
        cy["sent_v1"] = cy["opportunity_pct"] - cy["risk_pct"]
        cy["sent_v2"] = (cy["opportunity_pct"] +
                         cy["neutral_pct"]) - cy["risk_pct"]

        # YoY deltas
        cy = cy.sort_values(["company_id", "year"])
        cy["talk_score_delta"] = cy.groupby("company_id")["talk_score"].diff()

        self.report_df = df
        self.cy_df = cy

        print(f"✅ Loaded: {len(df)} reports, {cy['company'].nunique()} companies, "
              f"{cy['year'].min()}-{cy['year'].max()}")

        return df, cy

    # -------------------------------------------------------------------------
    # CSV Exports
    # -------------------------------------------------------------------------

    def export_csvs(self):
        """Export all CSV files."""
        if self.cy_df is None:
            self.load_data()

        print(f"\n{'='*60}")
        print("EXPORTING CSV FILES")
        print(f"{'='*60}")

        cy = self.cy_df

        # 1. Company-Year
        cy_export = cy[[
            "company", "company_id", "year", "is_translated",
            "total_chunks", "climate_chunks", "climate_pct",
            "netzero_chunks", "netzero_pct",
            "specificity_w", "commitment_w", "talk_score", "talk_score_delta",
            "opportunity_pct", "risk_pct", "neutral_pct", "sent_v1", "sent_v2"
        ]].copy()
        cy_export.to_csv(self.output_dir / "company_year.csv", index=False)
        print(f"   ✓ company_year.csv ({len(cy_export)} rows)")

        # 2. Company Totals
        company_totals = cy.groupby(["company_id", "company"]).agg({
            "year": ["min", "max", "count"],
            "total_chunks": "sum",
            "climate_chunks": "sum",
            "climate_pct": "mean",
            "netzero_chunks": "sum",
            "netzero_pct": "mean",
            "specificity_w": "mean",
            "commitment_w": "mean",
            "talk_score": ["mean", "std", "first", "last"],
            "talk_score_delta": "mean",
            "opportunity_pct": "mean",
            "risk_pct": "mean",
            "neutral_pct": "mean",
            "sent_v2": ["mean", "first", "last"],
        }).reset_index()
        company_totals.columns = [
            "company_id", "company",
            "year_min", "year_max", "n_years",
            "total_chunks", "climate_chunks", "climate_pct_avg",
            "netzero_chunks", "netzero_pct_avg",
            "specificity_avg", "commitment_avg",
            "talk_score_avg", "talk_score_std", "talk_score_first", "talk_score_last",
            "talk_score_delta_avg",
            "opportunity_pct_avg", "risk_pct_avg", "neutral_pct_avg",
            "sent_v2_avg", "sent_v2_first", "sent_v2_last"
        ]
        company_totals["talk_score_trend"] = (
            company_totals["talk_score_last"] -
            company_totals["talk_score_first"]
        )
        company_totals["sent_v2_trend"] = (
            company_totals["sent_v2_last"] - company_totals["sent_v2_first"]
        )
        company_totals.to_csv(
            self.output_dir / "company_totals.csv", index=False)
        print(f"   ✓ company_totals.csv ({len(company_totals)} companies)")

        # 3. Yearly Industry
        yearly = cy.groupby("year").agg({
            "company_id": "nunique",
            "total_chunks": "sum",
            "climate_chunks": "sum",
            "climate_pct": ["mean", "std"],
            "netzero_chunks": "sum",
            "netzero_pct": ["mean", "std"],
            "specificity_w": ["mean", "std"],
            "commitment_w": ["mean", "std"],
            "talk_score": ["mean", "std", "min", "max"],
            "talk_score_delta": "mean",
            "opportunity_pct": "mean",
            "risk_pct": "mean",
            "neutral_pct": "mean",
            "sent_v2": ["mean", "std"],
        }).reset_index()
        yearly.columns = [
            "year", "n_companies",
            "total_chunks", "climate_chunks",
            "climate_pct_mean", "climate_pct_std",
            "netzero_chunks", "netzero_pct_mean", "netzero_pct_std",
            "specificity_mean", "specificity_std",
            "commitment_mean", "commitment_std",
            "talk_score_mean", "talk_score_std", "talk_score_min", "talk_score_max",
            "talk_score_delta_mean",
            "opportunity_pct", "risk_pct", "neutral_pct",
            "sent_v2_mean", "sent_v2_std"
        ]
        yearly.to_csv(self.output_dir / "yearly_industry.csv", index=False)
        print(f"   ✓ yearly_industry.csv ({len(yearly)} years)")

        # 4. Funnel
        funnel_cy = cy[[
            "company", "company_id", "year",
            "total_chunks", "climate_chunks", "climate_pct",
            "netzero_chunks", "netzero_pct"
        ]].copy()
        funnel_cy["netzero_of_climate_pct"] = np.where(
            funnel_cy["climate_chunks"] > 0,
            funnel_cy["netzero_chunks"] / funnel_cy["climate_chunks"] * 100,
            0
        )
        funnel_cy.to_csv(self.output_dir /
                         "funnel_company_year.csv", index=False)
        print(f"   ✓ funnel_company_year.csv ({len(funnel_cy)} rows)")

        print(f"\n   📁 All CSVs saved to: {self.output_dir}/")

    # -------------------------------------------------------------------------
    # Plot Generation
    # -------------------------------------------------------------------------

    def _get_yearly_plot_data(self) -> pd.DataFrame:
        """Get aggregated yearly data for plotting."""
        return self.cy_df.groupby("year").agg({
            "climate_pct": "mean",
            "specificity_w": "mean",
            "commitment_w": "mean",
            "talk_score": ["mean", "std"],
            "talk_score_delta": "mean",
            "opportunity_pct": "mean",
            "risk_pct": "mean",
            "neutral_pct": "mean",
            "netzero_pct": "mean",
            "company_id": "nunique",
        }).reset_index()

    def plot_main_slide(self):
        """Plot: Main slide - Volume, Specificity, Commitment, Net-Zero."""
        yearly = self._get_yearly_plot_data()
        yearly.columns = [
            "year", "volume", "specificity", "commitment",
            "talk_mean", "talk_std", "talk_delta",
            "opp", "risk", "neutral", "netzero", "n_companies"
        ]

        fig, ax = plt.subplots(figsize=self.config.figsize_large)
        ax.plot(yearly["year"], yearly["volume"], "#7B1FA2", lw=3.5,
                marker="o", ms=11, label="Volume (% climate)")
        ax.plot(yearly["year"], yearly["netzero"], "#E91E63", lw=3.5,
                marker="D", ms=10, label="Net-Zero Focus (%)", linestyle="--")
        ax.plot(yearly["year"], yearly["specificity"] * 100, "#1565C0",
                lw=3.5, marker="s", ms=11, label="Specificity")
        ax.plot(yearly["year"], yearly["commitment"] * 100, "#2E7D32",
                lw=3.5, marker="^", ms=11, label="Commitment")
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel("Score (0-100)", fontsize=14)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.set_title("Climate Reporting in the Steel Industry",
                     fontsize=18, fontweight="bold")
        ax.legend(loc="upper left", fontsize=11, frameon=True, fancybox=True,
                  shadow=True, facecolor="#F5F5F5", edgecolor="gray")
        self._save_plot("slide_main.png")

    def plot_sentiment_trend(self):
        """Plot 2: Sentiment trend."""
        yearly = self._get_yearly_plot_data()
        yearly.columns = [
            "year", "volume", "specificity", "commitment",
            "talk_mean", "talk_std", "talk_delta",
            "opp", "risk", "neutral", "netzero", "n_companies"
        ]

        fig, ax = plt.subplots(figsize=self.config.figsize_medium)
        ax.plot(yearly["year"], yearly["opp"], "g-o",
                lw=3, ms=9, label="Opportunity")
        ax.plot(yearly["year"], yearly["neutral"], "gray", ls="-", marker="s",
                lw=3, ms=9, label="Neutral")
        ax.plot(yearly["year"], yearly["risk"],
                "r-^", lw=3, ms=9, label="Risk")
        ax.set_xlabel("Year", fontsize=13)
        ax.set_ylabel("% of Climate Chunks", fontsize=13)
        ax.set_title("Sentiment Trend", fontsize=16, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=11, frameon=True, fancybox=True,
                  shadow=True, facecolor="#F5F5F5", edgecolor="gray")
        self._save_plot("slide_sentiment_trend.png")

    def plot_talk_score_trend(self):
        """Plot 3: Talk score trend + delta."""
        yearly = self._get_yearly_plot_data()
        yearly.columns = [
            "year", "volume", "specificity", "commitment",
            "talk_mean", "talk_std", "talk_delta",
            "opp", "risk", "neutral", "netzero", "n_companies"
        ]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax1 = axes[0]
        ax1.plot(yearly["year"], yearly["talk_mean"],
                 "#1976D2", lw=3, marker="o", ms=10)
        ax1.fill_between(yearly["year"],
                         yearly["talk_mean"] - yearly["talk_std"],
                         yearly["talk_mean"] + yearly["talk_std"], alpha=0.2)
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Talk Score (0-100)")
        ax1.set_title("Talk Score Trend\n(20% volume + 40% specificity + 40% commitment)",
                      fontweight="bold")
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        colors = ["green" if d >
                  0 else "red" for d in yearly["talk_delta"].fillna(0)]
        ax2.bar(yearly["year"], yearly["talk_delta"].fillna(0), color=colors,
                alpha=0.7, edgecolor="white")
        ax2.axhline(0, color="black", lw=1)
        ax2.set_xlabel("Year")
        ax2.set_ylabel("YoY Change")
        ax2.set_title("Talk Score YoY Delta", fontweight="bold")
        ax2.grid(True, alpha=0.3)
        self._save_plot("talk_score_trend.png")

    def plot_funnel_trend(self):
        """Plot 4: Funnel plot."""
        yearly = self._get_yearly_plot_data()
        yearly.columns = [
            "year", "volume", "specificity", "commitment",
            "talk_mean", "talk_std", "talk_delta",
            "opp", "risk", "neutral", "netzero", "n_companies"
        ]

        fig, ax = plt.subplots(figsize=self.config.figsize_medium)
        ax.plot(yearly["year"], yearly["volume"], "#7B1FA2", lw=3,
                marker="o", ms=10, label="Climate % of Total")
        ax.plot(yearly["year"], yearly["netzero"], "#FF6F00", lw=3,
                marker="^", ms=10, label="NetZero % of Climate")
        ax.set_xlabel("Year", fontsize=13)
        ax.set_ylabel("Percentage", fontsize=13)
        ax.set_title("Content Funnel: Total → Climate → NetZero",
                     fontsize=16, fontweight="bold")
        ax.set_ylim(0, max(yearly["volume"].max(),
                    yearly["netzero"].max()) * 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=11, frameon=True)
        self._save_plot("funnel_trend.png")

    def plot_talk_score_per_company(self):
        """Plot 5: Talk score per company."""
        company_stats = self.cy_df.groupby("company").agg({
            "talk_score": ["mean", "last"],
            "talk_score_delta": "mean",
            "year": "count"
        }).reset_index()
        company_stats.columns = ["company", "talk_mean", "talk_latest",
                                 "talk_delta_avg", "n_years"]
        company_stats = company_stats.sort_values("talk_latest")

        fig, ax = plt.subplots(figsize=(10, max(6, len(company_stats) * 0.4)))
        colors = plt.cm.RdYlGn(company_stats["talk_latest"] / 100)
        ax.barh(company_stats["company"], company_stats["talk_latest"],
                color=colors, edgecolor="white")
        for i, (_, r) in enumerate(company_stats.iterrows()):
            ax.text(r["talk_latest"] + 1, i, f'{r["talk_latest"]:.0f}',
                    va="center", fontsize=9)
        ax.set_xlabel("Talk Score (Latest Year)")
        ax.set_xlim(0, 105)
        ax.set_title("Talk Score by Company", fontweight="bold")
        self._save_plot("talk_score_per_company.png")

    def plot_per_company_components(self):
        """Plot 6: Per-company components."""
        companies = sorted(self.cy_df["company"].unique())
        n_cols, n_rows = 3, int(np.ceil(len(companies) / 3))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                                 squeeze=False)
        axes = axes.flatten()

        for i, company in enumerate(companies):
            ax = axes[i]
            cd = self.cy_df[self.cy_df["company"]
                            == company].sort_values("year")
            ax.bar(cd["year"], cd["climate_pct"], color="lightgray",
                   alpha=0.5, label="Volume", width=0.7)
            ax.plot(cd["year"], cd["specificity_w"] * 100, "b-s", ms=5,
                    lw=1.5, label="Specificity")
            ax.plot(cd["year"], cd["commitment_w"] * 100, "g-^", ms=5,
                    lw=1.5, label="Commitment")
            ax.plot(cd["year"], cd["talk_score"], "k-o", ms=6, lw=2,
                    label="Talk Score")
            ax.set_title(company, fontweight="bold", fontsize=10)
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=7, loc="upper left")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Per-Company: Components + Talk Score",
                     fontsize=14, fontweight="bold", y=1.01)
        self._save_plot("per_company_components.png")

    def plot_per_company_sentiment(self):
        """Plot 7: Per-company sentiment stacked area."""
        companies = sorted(self.cy_df["company"].unique())
        n_cols, n_rows = 3, int(np.ceil(len(companies) / 3))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                                 squeeze=False)
        axes = axes.flatten()

        for i, company in enumerate(companies):
            ax = axes[i]
            cd = self.cy_df[self.cy_df["company"]
                            == company].sort_values("year")

            ax.fill_between(cd["year"], 0, cd["opportunity_pct"],
                            color="green", alpha=0.7, label="Opportunity")
            ax.fill_between(cd["year"], cd["opportunity_pct"],
                            cd["opportunity_pct"] + cd["neutral_pct"],
                            color="gray", alpha=0.7, label="Neutral")
            ax.fill_between(cd["year"],
                            cd["opportunity_pct"] + cd["neutral_pct"], 100,
                            color="red", alpha=0.7, label="Risk")

            ax.set_title(company, fontweight="bold", fontsize=10)
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=7, loc="upper right")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Per-Company Sentiment Composition",
                     fontsize=14, fontweight="bold", y=1.01)
        self._save_plot("per_company_sentiment.png")

    def plot_sentiment_all_companies(self):
        """Plot 8: All companies sentiment trajectory."""
        companies = sorted(self.cy_df["company"].unique())

        fig, ax = plt.subplots(figsize=(14, 7))
        colors = plt.cm.tab20(np.linspace(0, 1, len(companies)))

        for i, company in enumerate(companies):
            cd = self.cy_df[self.cy_df["company"]
                            == company].sort_values("year")
            ax.plot(cd["year"], cd["sent_v2"], "-o", color=colors[i],
                    lw=1.5, ms=5, alpha=0.6)
            if len(cd) > 0:
                ax.annotate(company, (cd["year"].iloc[-1] + 0.1, cd["sent_v2"].iloc[-1]),
                            fontsize=7, color=colors[i])

        # Industry average
        yearly_sent = self.cy_df.groupby("year")["sent_v2"].mean()
        ax.plot(yearly_sent.index, yearly_sent.values, "k-", lw=4,
                label="Industry Avg", zorder=10)
        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Sentiment: (Opp+Neutral) - Risk", fontsize=12)
        ax.set_title("Sentiment Trajectory by Company",
                     fontsize=14, fontweight="bold")
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 1.5)
        ax.grid(True, alpha=0.3)
        self._save_plot("sentiment_all_companies.png")

    def plot_netzero_funnel(self):
        """Plot N0-1: The funnel visualization."""
        if self.cy_df["netzero_chunks"].sum() == 0:
            print("   ⚠️ No net-zero data, skipping funnel plot")
            return

        yearly = self.cy_df.groupby("year").agg({
            "total_chunks": "sum",
            "climate_chunks": "sum",
            "netzero_chunks": "sum",
        }).reset_index()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Stacked bar
        ax1 = axes[0]
        years = yearly["year"]
        non_climate = yearly["total_chunks"] - yearly["climate_chunks"]
        climate_non_n0 = yearly["climate_chunks"] - yearly["netzero_chunks"]
        n0 = yearly["netzero_chunks"]

        ax1.bar(years, non_climate, label="Non-Climate",
                color="#E0E0E0", edgecolor="white")
        ax1.bar(years, climate_non_n0, bottom=non_climate, label="Climate (General)",
                color="#90CAF9", edgecolor="white")
        ax1.bar(years, n0, bottom=non_climate + climate_non_n0, label="Net-Zero Focus",
                color="#1565C0", edgecolor="white")
        ax1.set_xlabel("Year", fontsize=12)
        ax1.set_ylabel("Number of Chunks", fontsize=12)
        ax1.set_title("Content Composition: The Funnel",
                      fontsize=14, fontweight="bold")
        ax1.legend(loc="upper left", fontsize=10)
        ax1.grid(True, alpha=0.3, axis="y")

        # Right: Percentages
        ax2 = axes[1]
        total = yearly["total_chunks"]
        climate_pct = yearly["climate_chunks"] / total * 100
        n0_pct_of_total = yearly["netzero_chunks"] / total * 100
        n0_pct_of_climate = np.where(
            yearly["climate_chunks"] > 0,
            yearly["netzero_chunks"] / yearly["climate_chunks"] * 100,
            0
        )

        ax2.plot(years, climate_pct, "b-o", lw=3,
                 ms=10, label="Climate % of Total")
        ax2.plot(years, n0_pct_of_total, "g-s", lw=3,
                 ms=10, label="Net-Zero % of Total")
        ax2.plot(years, n0_pct_of_climate, "r-^", lw=2, ms=8, alpha=0.7,
                 label="Net-Zero % of Climate", linestyle="--")
        ax2.set_xlabel("Year", fontsize=12)
        ax2.set_ylabel("Percentage", fontsize=12)
        ax2.set_title("Focus Ratios Over Time", fontsize=14, fontweight="bold")
        ax2.legend(loc="upper left", fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, max(climate_pct.max() * 1.1, 100))

        self._save_plot("n0_funnel.png")

    def plot_netzero_quality_comparison(self):
        """Plot N0-3: Quality comparison N0 vs general."""
        if self.cy_df["netzero_chunks"].sum() == 0:
            print("   ⚠️ No net-zero data, skipping quality comparison plot")
            return

        yearly = self.cy_df.groupby("year").agg({
            "specificity_w": "mean",
            "commitment_w": "mean",
            "n0_specificity": "mean",
            "n0_commitment": "mean",
            "non_n0_specificity": "mean",
            "non_n0_commitment": "mean",
        }).reset_index()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax1 = axes[0]
        ax1.plot(yearly["year"], yearly["n0_specificity"] * 100, "b-o", lw=3, ms=9,
                 label="Net-Zero Chunks")
        ax1.plot(yearly["year"], yearly["non_n0_specificity"] * 100, "gray", ls="--",
                 marker="s", lw=2, ms=7, alpha=0.7, label="General Climate Chunks")
        ax1.set_xlabel("Year", fontsize=12)
        ax1.set_ylabel("Specificity Score", fontsize=12)
        ax1.set_title("Specificity: Net-Zero vs General Climate",
                      fontsize=13, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)

        ax2 = axes[1]
        ax2.plot(yearly["year"], yearly["n0_commitment"] * 100, "g-o", lw=3, ms=9,
                 label="Net-Zero Chunks")
        ax2.plot(yearly["year"], yearly["non_n0_commitment"] * 100, "gray", ls="--",
                 marker="s", lw=2, ms=7, alpha=0.7, label="General Climate Chunks")
        ax2.set_xlabel("Year", fontsize=12)
        ax2.set_ylabel("Commitment Score", fontsize=12)
        ax2.set_title("Commitment: Net-Zero vs General Climate",
                      fontsize=13, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)

        self._save_plot("n0_quality_comparison.png")

    def plot_netzero_per_company(self):
        """Plot N0-4: Per-company net-zero focus."""
        if self.cy_df["netzero_chunks"].sum() == 0:
            print("   ⚠️ No net-zero data, skipping per-company N0 plot")
            return

        company_n0 = self.cy_df.groupby("company").agg({
            "netzero_pct": ["mean", "last"],
            "climate_pct": "mean",
            "netzero_chunks": "sum",
            "climate_chunks": "sum",
        }).reset_index()
        company_n0.columns = ["company", "n0_pct_mean", "n0_pct_latest",
                              "climate_pct", "n0_total", "climate_total"]
        company_n0["n0_of_climate"] = np.where(
            company_n0["climate_total"] > 0,
            company_n0["n0_total"] / company_n0["climate_total"] * 100,
            0
        )
        company_n0 = company_n0.sort_values("n0_pct_latest")

        fig, axes = plt.subplots(1, 2, figsize=(
            14, max(6, len(company_n0) * 0.4)))

        ax1 = axes[0]
        max_val = max(company_n0["n0_pct_latest"].max(), 1)
        colors = plt.cm.Blues(company_n0["n0_pct_latest"] / max_val)
        ax1.barh(company_n0["company"], company_n0["n0_pct_latest"],
                 color=colors, edgecolor="white")
        for i, (_, r) in enumerate(company_n0.iterrows()):
            ax1.text(r["n0_pct_latest"] + 0.3, i, f'{r["n0_pct_latest"]:.1f}%',
                     va="center", fontsize=9)
        ax1.set_xlabel("Net-Zero % of Report (Latest Year)")
        ax1.set_title("Net-Zero Focus by Company", fontweight="bold")
        ax1.set_xlim(0, company_n0["n0_pct_latest"].max() * 1.2)

        ax2 = axes[1]
        max_val2 = max(company_n0["n0_of_climate"].max(), 1)
        colors2 = plt.cm.Greens(company_n0["n0_of_climate"] / max_val2)
        ax2.barh(company_n0["company"], company_n0["n0_of_climate"],
                 color=colors2, edgecolor="white")
        for i, (_, r) in enumerate(company_n0.iterrows()):
            ax2.text(r["n0_of_climate"] + 0.5, i, f'{r["n0_of_climate"]:.1f}%',
                     va="center", fontsize=9)
        ax2.set_xlabel("Net-Zero % of Climate Content")
        ax2.set_title("Net-Zero Depth (of Climate Talk)", fontweight="bold")
        ax2.set_xlim(0, company_n0["n0_of_climate"].max() * 1.2)

        self._save_plot("n0_per_company.png")

    def plot_netzero_gap(self):
        """Plot N0-6: The gap between climate talk and N0 action."""
        if self.cy_df["netzero_chunks"].sum() == 0:
            print("   ⚠️ No net-zero data, skipping gap plot")
            return

        yearly = self.cy_df.groupby("year").agg({
            "climate_pct": "mean",
            "netzero_pct": "mean",
        }).reset_index()

        fig, ax = plt.subplots(figsize=self.config.figsize_medium)

        years = yearly["year"]
        climate = yearly["climate_pct"]
        n0 = yearly["netzero_pct"]

        ax.fill_between(years, n0, climate, color="#FFCDD2",
                        alpha=0.7, label="The Gap")
        ax.plot(years, climate, "#7B1FA2", lw=4,
                marker="o", ms=12, label="Climate Talk")
        ax.plot(years, n0, "#1565C0", lw=4, marker="s",
                ms=12, label="Net-Zero Focus")

        # Annotate gap
        mid_idx = len(years) // 2
        mid_year = years.iloc[mid_idx]
        mid_climate = climate.iloc[mid_idx]
        mid_n0 = n0.iloc[mid_idx]
        ax.annotate("", xy=(mid_year, mid_n0), xytext=(mid_year, mid_climate),
                    arrowprops=dict(arrowstyle="<->", color="red", lw=2))
        ax.text(mid_year + 0.3, (mid_climate + mid_n0) / 2,
                f"Gap:\n{mid_climate - mid_n0:.1f}%",
                fontsize=11, color="red", fontweight="bold")

        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel("% of Report Content", fontsize=14)
        ax.set_title("The Gap: Climate Talk vs Net-Zero Focus",
                     fontsize=16, fontweight="bold")
        ax.legend(loc="upper left", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(climate.max() * 1.2, 50))

        self._save_plot("n0_gap_analysis.png")

    def generate_all_plots(self):
        """Generate all plots."""
        if self.cy_df is None:
            self.load_data()

        print(f"\n{'='*60}")
        print("GENERATING PLOTS")
        print(f"{'='*60}")

        self.plot_main_slide()
        self.plot_sentiment_trend()
        self.plot_talk_score_trend()
        self.plot_funnel_trend()
        self.plot_talk_score_per_company()
        self.plot_per_company_components()
        self.plot_per_company_sentiment()
        self.plot_sentiment_all_companies()

        # Net-zero plots
        self.plot_netzero_funnel()
        self.plot_netzero_quality_comparison()
        self.plot_netzero_per_company()
        self.plot_netzero_gap()

        print(f"\n   📁 All plots saved to: {self.output_dir}/")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def print_summary(self):
        """Print summary statistics."""
        if self.cy_df is None:
            self.load_data()

        cy = self.cy_df
        n_reports = len(self.report_df)
        n_chunks = int(cy["climate_chunks"].sum())
        n_companies = cy["company"].nunique()
        year_range = f"{cy['year'].min()}-{cy['year'].max()}"

        total_n0 = int(cy["netzero_chunks"].sum())
        n0_ratio = total_n0 / n_chunks * 100 if n_chunks > 0 else 0

        print(f"""
{'='*60}
📊 SUMMARY
{'='*60}
Data: {n_reports} reports · {n_chunks:,} chunks · {n_companies} companies · {year_range}
Method: ClimateBERT (scores weighted by detector confidence)

Talk Score = 20% volume + 40% specificity + 40% commitment

Net-Zero Analysis:
  Climate chunks: {n_chunks:,}
  Net-Zero chunks: {total_n0:,} ({n0_ratio:.1f}% of climate content)
{'='*60}
""")

    # -------------------------------------------------------------------------
    # Word Cloud & Frequency Analysis
    # -------------------------------------------------------------------------

    def _load_chunk_texts(self) -> Tuple[List[str], Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
        """Load all chunk texts grouped by category."""
        bert_files = list(self.cache_dir.glob("*_bert.json"))

        all_texts = []
        texts_by_sentiment = {"opportunity": [], "risk": [], "neutral": []}
        texts_by_commitment = {"yes": [], "no": []}
        texts_by_netzero = {"net-zero": [], "reduction": [], "none": []}

        for fp in bert_files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)

                year = data.get("year", 0)
                if self.config.exclude_year_gte and year >= self.config.exclude_year_gte:
                    continue

                for chunk in data.get("chunks", []):
                    text = chunk.get("text", "")
                    if not text:
                        continue

                    all_texts.append(text)

                    # By sentiment
                    sent = chunk.get("sentiment_label", "neutral")
                    if sent in texts_by_sentiment:
                        texts_by_sentiment[sent].append(text)

                    # By commitment
                    comm = chunk.get("commitment_label", "no")
                    if comm in texts_by_commitment:
                        texts_by_commitment[comm].append(text)

                    # By netzero
                    n0 = chunk.get("netzero_label", "none")
                    if n0 in texts_by_netzero:
                        texts_by_netzero[n0].append(text)

            except Exception:
                pass

        return all_texts, texts_by_sentiment, texts_by_commitment, texts_by_netzero

    def generate_wordclouds(self):
        """Generate word cloud visualizations."""
        try:
            from wordcloud import WordCloud
        except ImportError:
            print("   ⚠️ wordcloud not installed. Run: pip install wordcloud")
            return

        print("\n   Generating word clouds...")
        all_texts, texts_by_sentiment, texts_by_commitment, texts_by_netzero = self._load_chunk_texts()

        # Main word clouds (sentiment + commitment)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        def make_cloud(ax, texts, title, colormap="viridis"):
            freq = get_word_frequencies(texts, top_n=100)
            if freq:
                wc = WordCloud(
                    width=800, height=400, background_color="white",
                    colormap=colormap, max_words=100
                )
                wc.generate_from_frequencies(dict(freq))
                ax.imshow(wc, interpolation="bilinear")
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.axis("off")

        make_cloud(axes[0, 0], all_texts, "📊 All Climate Chunks", "viridis")
        make_cloud(axes[0, 1], texts_by_sentiment["opportunity"],
                   "🌱 Opportunity Sentiment", "Greens")
        make_cloud(axes[1, 0], texts_by_sentiment["risk"],
                   "⚠️ Risk Sentiment", "Reds")
        make_cloud(axes[1, 1], texts_by_commitment["yes"],
                   "🤝 Commitment Language", "Blues")

        plt.tight_layout()
        self._save_plot("wordclouds.png")

        # Net-zero word clouds
        if sum(len(t) for t in texts_by_netzero.values()) > 0:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            make_cloud(axes[0], texts_by_netzero["net-zero"],
                       "🎯 Net-Zero Targets", "Blues")
            make_cloud(axes[1], texts_by_netzero["reduction"],
                       "📉 Reduction Targets", "Greens")
            make_cloud(axes[2], texts_by_netzero["none"],
                       "⬜ Climate (No Target)", "Greys")
            plt.tight_layout()
            self._save_plot("n0_wordclouds.png")

    def generate_word_frequency_plots(self):
        """Generate word frequency bar charts."""
        print("\n   Generating word frequency plots...")
        all_texts, texts_by_sentiment, texts_by_commitment, texts_by_netzero = self._load_chunk_texts()

        # Get frequencies
        freq_all = get_word_frequencies(all_texts, top_n=30)
        freq_opp = get_word_frequencies(
            texts_by_sentiment["opportunity"], top_n=20)
        freq_risk = get_word_frequencies(texts_by_sentiment["risk"], top_n=20)
        freq_commit = get_word_frequencies(
            texts_by_commitment["yes"], top_n=20)

        # Print to console
        print("\n   📊 ALL CHUNKS (top 30):")
        print("   " + ", ".join([f"{w}({c})" for w, c in freq_all[:15]]))

        print("\n   🌱 OPPORTUNITY chunks (top 15):")
        print("   " + ", ".join([f"{w}({c})" for w, c in freq_opp[:15]]))

        print("\n   ⚠️ RISK chunks (top 15):")
        print("   " + ", ".join([f"{w}({c})" for w, c in freq_risk[:15]]))

        # Bar chart
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        def plot_freq_bars(ax, freqs, title, color):
            if not freqs:
                return
            words, counts = zip(*freqs[:15])
            y_pos = range(len(words))
            ax.barh(y_pos, counts, color=color, alpha=0.7, edgecolor="white")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words)
            ax.invert_yaxis()
            ax.set_xlabel("Frequency")
            ax.set_title(title, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x")

        plot_freq_bars(axes[0, 0], freq_all[:15],
                       "📊 All Chunks - Top Words", "#1976D2")
        plot_freq_bars(axes[0, 1], freq_opp[:15],
                       "🌱 Opportunity - Top Words", "#4CAF50")
        plot_freq_bars(axes[1, 0], freq_risk[:15],
                       "⚠️ Risk - Top Words", "#F44336")
        plot_freq_bars(axes[1, 1], freq_commit[:15],
                       "🤝 Commitment - Top Words", "#7B1FA2")

        plt.tight_layout()
        self._save_plot("word_frequencies.png")

        # Net-zero frequencies
        freq_netzero = get_word_frequencies(
            texts_by_netzero["net-zero"], top_n=20)
        freq_reduction = get_word_frequencies(
            texts_by_netzero["reduction"], top_n=20)
        freq_none = get_word_frequencies(texts_by_netzero["none"], top_n=20)

        if freq_reduction:
            fig, axes = plt.subplots(1, 3, figsize=(16, 6))
            plot_freq_bars(axes[0], freq_netzero[:12], "🎯 Net-Zero", "#1565C0")
            plot_freq_bars(axes[1], freq_reduction[:12],
                           "📉 Reduction", "#2E7D32")
            plot_freq_bars(axes[2], freq_none[:12], "⬜ No Target", "#757575")
            plt.tight_layout()
            self._save_plot("n0_word_frequencies.png")

    # -------------------------------------------------------------------------
    # Chunk Analyzer
    # -------------------------------------------------------------------------

    def analyze_chunks(
        self,
        n_samples: int = 10,
        label_filter: Optional[str] = None,
        filter_type: str = "netzero",
        random_seed: int = 42,
        show_scores: bool = True,
        export_path: Optional[str] = None,
    ) -> List[Dict]:
        """
        Analyze and inspect classified chunks.

        Args:
            n_samples: Number of random samples to display (default 10)
            label_filter: Filter value (e.g., 'reduction', 'opportunity', 'yes')
            filter_type: Type of filter - 'netzero', 'sentiment', or 'commitment'
            random_seed: For reproducibility
            show_scores: Show confidence scores
            export_path: If provided, export all matching chunks to this JSON file

        Returns:
            List of matching chunks

        Examples:
            # Browse net-zero chunks
            viz.analyze_chunks(n_samples=10, label_filter='reduction', filter_type='netzero')

            # Browse opportunity sentiment
            viz.analyze_chunks(n_samples=10, label_filter='opportunity', filter_type='sentiment')

            # Export all commitment chunks
            viz.analyze_chunks(label_filter='yes', filter_type='commitment', export_path='commits.json')
        """
        import random
        random.seed(random_seed)

        bert_files = list(self.cache_dir.glob("*_bert.json"))

        # Map filter_type to JSON key
        filter_key_map = {
            "netzero": "netzero_label",
            "sentiment": "sentiment_label",
            "commitment": "commitment_label",
            "specificity": "specificity_label",
        }
        filter_key = filter_key_map.get(filter_type, "netzero_label")

        all_chunks = []

        for fp in bert_files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)

                year = data.get("year", 0)
                if self.config.exclude_year_gte and year >= self.config.exclude_year_gte:
                    continue

                company = data.get("company", "Unknown")

                for chunk in data.get("chunks", []):
                    chunk_label = chunk.get(filter_key, "none")

                    if label_filter and chunk_label != label_filter:
                        continue

                    all_chunks.append({
                        "company": company,
                        "year": year,
                        # Truncate for display
                        "text": chunk.get("text", "")[:500],
                        "full_text": chunk.get("text", ""),
                        "netzero_label": chunk.get("netzero_label", "none"),
                        "netzero_score": chunk.get("netzero_score", 0),
                        "sentiment_label": chunk.get("sentiment_label", "neutral"),
                        "sentiment_score": chunk.get("sentiment_score", 0),
                        "commitment_label": chunk.get("commitment_label", "no"),
                        "commitment_score": chunk.get("commitment_score", 0),
                        "specificity_label": chunk.get("specificity_label", "non"),
                        "specificity_score": chunk.get("specificity_score", 0),
                        "detector_score": chunk.get("detector_score", 0),
                    })
            except Exception:
                pass

        # Export if requested
        if export_path:
            export_data = [{
                "company": c["company"],
                "year": c["year"],
                "text": c["full_text"],
                "netzero_label": c["netzero_label"],
                "netzero_score": c["netzero_score"],
                "sentiment_label": c["sentiment_label"],
                "commitment_label": c["commitment_label"],
            } for c in all_chunks]
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            print(f"✅ Exported {len(export_data)} chunks to {export_path}")

        # Summary stats
        label_counts = Counter(c[filter_key] for c in all_chunks)
        print(f"\n{'='*60}")
        print(f"CHUNK ANALYSIS")
        print(f"{'='*60}")
        print(f"Total chunks: {len(all_chunks):,}")
        print(f"Filter: {filter_type}={label_filter or 'all'}")
        print(f"\nLabel distribution ({filter_type}):")
        for label, count in sorted(label_counts.items()):
            pct = count / len(all_chunks) * 100 if all_chunks else 0
            print(f"   {label}: {count:,} ({pct:.1f}%)")

        # Score distribution for filtered label
        if label_filter and all_chunks:
            score_key = f"{filter_type}_score" if filter_type != "netzero" else "netzero_score"
            scores = [c.get(score_key, 0) for c in all_chunks]
            if scores:
                print(f"\nConfidence scores for '{label_filter}':")
                print(f"   Mean: {np.mean(scores):.3f}")
                print(f"   Median: {np.median(scores):.3f}")
                print(
                    f"   Min: {np.min(scores):.3f}, Max: {np.max(scores):.3f}")

        # Sample display
        if all_chunks and n_samples > 0:
            samples = random.sample(
                all_chunks, min(n_samples, len(all_chunks)))

            print(f"\n{'='*60}")
            print(f"SAMPLE CHUNKS (n={len(samples)})")
            print(f"{'='*60}")

            for i, chunk in enumerate(samples, 1):
                print(f"\n--- [{i}] {chunk['company']} ({chunk['year']}) ---")
                if show_scores:
                    print(f"N0: {chunk['netzero_label']} ({chunk['netzero_score']:.2f}) | "
                          f"Sent: {chunk['sentiment_label']} | "
                          f"Commit: {chunk['commitment_label']} | "
                          f"Spec: {chunk['specificity_label']}")
                else:
                    print(f"Labels: N0={chunk['netzero_label']}, "
                          f"Sent={chunk['sentiment_label']}, "
                          f"Commit={chunk['commitment_label']}")
                print(f"\n{chunk['text']}")
                if len(chunk["full_text"]) > 500:
                    print(
                        f"... [truncated, {len(chunk['full_text'])} chars total]")

        return all_chunks

    # -------------------------------------------------------------------------
    # Full Pipeline (updated)
    # -------------------------------------------------------------------------

    def run_all(self, include_wordclouds: bool = False, include_frequencies: bool = True):
        """
        Run complete visualization pipeline.

        Args:
            include_wordclouds: Generate word cloud plots (requires wordcloud package)
            include_frequencies: Generate word frequency analysis
        """
        self.load_data()
        self.export_csvs()
        self.generate_all_plots()

        if include_frequencies:
            self.generate_word_frequency_plots()

        if include_wordclouds:
            self.generate_wordclouds()

        self.print_summary()


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def visualize_results(
    cache_dir: str = "cache",
    output_dir: str = "out",
    exclude_year_gte: int = 2025,
    include_wordclouds: bool = True,
    include_frequencies: bool = True,
) -> ClimateBERTVisualizer:
    """
    Convenience function to generate all visualizations.

    Args:
        cache_dir: Directory with BERT cache files
        output_dir: Directory for output files
        exclude_year_gte: Exclude years >= this value
        include_wordclouds: Generate word clouds (requires: pip install wordcloud)
        include_frequencies: Generate word frequency analysis

    Returns:
        ClimateBERTVisualizer instance
    """
    config = VisualizationConfig(
        cache_dir=cache_dir,
        output_dir=output_dir,
        exclude_year_gte=exclude_year_gte,
    )
    viz = ClimateBERTVisualizer(config)
    viz.run_all(include_wordclouds=include_wordclouds,
                include_frequencies=include_frequencies)
    return viz


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("BERT Visualization Module")
    print("=" * 60)
    print("\nUsage:")
    print("  from bert_2 import visualize_results, ClimateBERTVisualizer")
    print()
    print("  # Quick start (after running bert_pipeline):")
    print("  visualize_results('cache', 'out')")
    print()
    print("  # Analyze specific chunks:")
    print("  viz = ClimateBERTVisualizer()")
    print("  viz.load_data()")
    print("  viz.analyze_chunks(n_samples=10, label_filter='reduction', filter_type='netzero')")
    print("  viz.analyze_chunks(n_samples=5, label_filter='opportunity', filter_type='sentiment')")
    print("  viz.analyze_chunks(label_filter='yes', filter_type='commitment', export_path='commits.json')")

    # visualize_results('cache', 'out', include_wordclouds=True)
