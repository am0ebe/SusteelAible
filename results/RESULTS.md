# SuSteelAible — Results

**Decarbonization in European Steel: What Emissions Data and 200 Sustainability Reports Tell Us**

This document summarizes the key findings from the SuSteelAible project. The analysis combines quantitative emissions data (13–16 companies, 2013–2024) with NLP analysis of ~200 corporate sustainability reports to understand technology lock-in, policy effectiveness, and the barriers and motivators companies articulate around decarbonization.

> **Interactive outputs** (topic maps, document explorer, barchart visualizations) are HTML files in `results/topics2/` — open them in a browser after cloning. The final presentation is at `results/final_presentation.pdf`. A "Talk vs Action" animation is at `results/models/talk_vs_action.mp4`.

---

## Table of Contents

1. [Key Findings at a Glance](#1-key-findings-at-a-glance)
2. [Emissions Analysis](#2-emissions-analysis)
3. [Modeling — What Drives Emissions?](#3-modeling--what-drives-emissions)
4. [NLP — What Companies Say](#4-nlp--what-companies-say)
5. [Data & Outputs](#5-data--outputs)

---

## 1. Key Findings at a Glance

| Finding | Detail |
|---------|--------|
| **Technology explains 80% of emission variance** | BF-BOF plants average 1.81 tCO₂e/t; EAF+EAF Stainless average 0.61 tCO₂e/t — a single-predictor decision tree achieves R²=0.798 |
| **Carbon prices rose 17×, sector intensity stayed flat** | EU ETS prices climbed ~€5 → €85+/tCO₂ (2013–2023); EU-wide steel intensity held at ~0.7–0.8 tCO₂e/t |
| **Policy interventions show no within-firm effect** | ETS, CBAM, and the Green Deal are not associated with statistically significant intensity reductions at firm level |
| **Coal prices Granger-cause intensity** | Coal price movements predict intensity changes with a ~2-year lag (p=0.0015); carbon price borderline (p=0.0018) |
| **Scope 2 declining for EAF only** | Grid decarbonization — not steel-process improvements — is the only visible downward trend; BF-BOF Scope 1 is structurally flat |
| **Action scores split entirely by technology** | EAF companies score 50–93 pts; BF-BOF companies score 12–40 pts; only 3 firms achieved the −2%/yr SBTi-aligned trend threshold |
| **Post-2020 improvement is real but narrow** | Outokumpu (+18.9 pts), Salzgitter (+14.9), SHS (+14.1), SSAB (+11.2) improved most; gains largely reflect tech roadmap announcements, not achieved emission reductions |
| **Barriers are market-side, not just technical** | Companies report limited availability of low-carbon steel inputs and certification costs more than technology constraints |
| **Motivators are commitment-driven** | Internal emission targets and low-carbon transition narratives dominate over regulatory compliance language |
| **Climate discourse is intensifying** | Talk scores trend upward across the corpus, especially post-2018; net-zero language spikes post-Paris Agreement |

---

## 2. Emissions Analysis

**Source:** `01_eda/EDA_emissions.ipynb`, `01_eda/external_drivers_eda_and_model.ipynb`
**Data:** 13 European steel companies with ≥4 years of data, Scope 1+2, 2013–2024
**Plots:** `results/EDA/`

### The Technology Gap

The single largest determinant of emission intensity is production technology. Scope 1 statistics across 103 company-year observations:

| Technology | N obs | Mean (tCO₂e/t) | Median | Std | Range |
|------------|-------|----------------|--------|-----|-------|
| BF-BOF | 63 | 1.737 | 1.765 | 0.258 | 1.383–2.541 |
| EAF Stainless | 18 | 0.439 | 0.430 | 0.053 | 0.352–0.531 |
| EAF (standard) | 22 | 0.100 | 0.091 | 0.020 | 0.080–0.131 |

BF-BOF plants use iron ore and coking coal — a chemically constrained process that cannot be incrementally improved away from fossil fuels. EAF plants melt scrap steel with electricity, which decarbonizes as the grid improves.

For total operational intensity (Scope 1+2), the two-group split (BF-BOF vs. all EAF) used in the baseline model yields: BF-BOF mean **1.814** tCO₂e/t, EAF mean **0.606** tCO₂e/t (N=103, spanning 2013–2024).

→ `results/EDA/01_boxplot_technology.png` — technology intensity distribution
→ `results/EDA/02_scope1_timeseries_european.png` — Scope 1 trends per company vs. EU/global benchmarks
→ `results/EDA/03_scope12_timeseries_european.png` — Scope 1+2 total intensity trends
→ `results/EDA/04_production_vs_intensity.png` — production scale vs. intensity (no scale effect within technology groups)
→ `results/EDA/05_scope2_by_company.png` — Scope 2 intensity by company

### The Carbon Price Paradox

EU ETS carbon prices increased 17-fold between 2013 and 2023 (from ~€5 to €85+/tCO₂). Over the same period, EU-wide steel emission intensity (EEA verified emissions) remained stable at ~0.7–0.8 tCO₂e/t.

This is not a market failure — it reflects technology lock-in. Existing BF-BOF plants cannot respond to price signals by switching production method. Decarbonization requires new capital investment in EAF or H₂-DRI capacity, not incremental efficiency gains.

→ `results/EDA/07_carbon_price_paradox.png`

### Scope 2 Progress

EAF companies show a visible downward trend in total operational intensity (Scope 1+2), driven by grid decarbonization rather than steel-process changes. BF-BOF Scope 1 emissions remain structurally flat across the full observation window.

### External Drivers — Panel Econometrics

Entity fixed-effects regressions (controlling for firm-level heterogeneity) show:

- **Only firm age** has a statistically significant within-firm effect on intensity
- **ETS, CBAM, Green Deal** show no significant within-firm intensity reduction
- **DiD analysis:** best treatment year 2019 (p=0.0375), effect disappears when controls added
- **Granger causality:** coal price Granger-causes intensity at lag 2 (p=0.0015); carbon price borderline (p=0.0018)

Technology-stratified FE models reveal different sensitivity patterns:
- **BF-BOF firms:** plant age dominates (coef=0.037, t=25, p<0.001); production scale also significant (coef=0.0008, p<0.01); ETS shows marginal negative effect (coef=−0.051, p<0.1)
- **EAF firms:** age effect present but weaker (coef=0.004, t=4.0, p<0.001); carbon price shows significant negative effect (coef=−0.0014, p<0.05)

Market signals (coal price, input costs) have more immediate causal traction on intensity than policy interventions, likely because they affect operating decisions within existing technology constraints. Technology transitions remain the only path to large reductions for BF-BOF operators.

---

## 3. Modeling — What Drives Emissions?

**Source:** `02_models/baseline_model.ipynb`, `02_models/predictions.ipynb`, `02_models/action_score_concept.ipynb`, `02_models/action_score_temporal.ipynb`
**Data:** `results/models/`

### Technology as Baseline

A single-feature decision tree using technology group (EAF vs. BF-BOF) as the only predictor, fitted on 103 company-year observations (2013–2024):

| Metric | Value |
|--------|-------|
| R² | **0.798** — technology explains ~80% of variance |
| MAE | 0.238 tCO₂e/t |
| RMSE | 0.296 tCO₂e/t |
| t-test (EAF vs BF-BOF) | t = −19.96, p < 0.001 |

This sets a hard ceiling: additional factors (pricing, policy, corporate commitments) can explain at most the remaining ~20% of variance within the current technology mix.

→ `results/models/baseline_tech_comparison.png`
→ `results/models/baseline_model_dataset.csv` — full dataset with predictions and residuals

### Action Score — Decarbonization Readiness (2024)

A 100-point composite score assessing current performance and trajectory per company:

| Dimension | Weight | Measure |
|-----------|--------|---------|
| Performance | 30 pts | Current intensity vs. 2.0 tCO₂e/t benchmark |
| Trend | 30 pts | Annual intensity improvement rate (threshold: −2%/yr, SBTi-aligned; R²≥0.5 required) |
| Data Quality | 15 pts | Time-series completeness + reporting coverage |
| Technology | 20 pts | Current tech and stated transformation plans (0 = no plans, 20 = clean tech at scale) |
| Renewable | 5 pts | Renewable electricity procurement % (EAF companies only) |

**2024 Results — all 13 companies:**

| Company | Technology | Intensity (tCO₂e/t) | Trend (%/yr) | Renewable % | Total Score | Category |
|---------|------------|---------------------|--------------|-------------|-------------|----------|
| Outokumpu | EAF Stainless | 0.50 | −4.26 | 89% | **92.5** | LEADER |
| SIDENOR Group | EAF | 0.33 | −3.53 | 18% | **79.0** | LEADER |
| Acerinox EU | EAF Stainless | 1.11 | −3.02 | 44% | **72.3** | LEADER |
| Feralpi Group | EAF | 0.29 | −0.83 | 35% | **69.1** | LEADER |
| Celsa Group | EAF | 0.22 | −0.09 | 1% | **50.7** | STRONG |
| SSAB | BF-BOF | 1.46 | +0.04 | — | **40.1** | MODERATE |
| Salzgitter AG | BF-BOF | 1.63 | −0.20 | — | **34.6** | MODERATE |
| ArcelorMittal | BF-BOF + EAF | 1.87 | −0.43 | — | **32.0** | MODERATE |
| SHS Group | BF-BOF | 1.76 | +0.36 | — | **28.6** | WEAK |
| Tata Steel Nederland | BF-BOF | 1.75 | +0.53 | — | **24.8** | WEAK |
| Voestalpine | BF-BOF | 1.84 | −1.43 | — | **24.4** | WEAK |
| Acciaierie d'Italia | BF-BOF | 2.59 | +4.09 | — | **19.0** | WEAK |
| Tata Steel UK | BF-BOF | 1.91 | −1.55 | — | **12.4** | WEAK |

Key patterns:
- **Technology divide is absolute:** all LEADER/STRONG companies are EAF; all MODERATE/WEAK are BF-BOF
- **Only 3 companies** achieved the −2%/yr SBTi-aligned trend threshold: Outokumpu (−4.26%/yr), SIDENOR (−3.53%/yr), Acerinox EU (−3.02%/yr) — all EAF, all post-2020 data only
- **Tata Steel UK scores 0 on Technology** — the only company with no stated transformation plan ("on hold")
- **SSAB, Salzgitter, Voestalpine** score highest on Technology among BF-BOF companies (HYBRIT, SALCOS, greentec steel programmes respectively), but their emission trends remain flat
- **Feralpi** achieves the lowest absolute intensity among non-stainless EAF (0.29 tCO₂e/t) with 12 years of consistent data

→ `results/models/action_scores_total.csv`
→ `results/models/action_score_breakdown.png`

### Action Score — Pre/Post-COVID Comparison

Comparing periods 2013–2019 (pre-COVID) vs. 2020–2024 (post-COVID) for companies with ≥3 years in each window:

| Company | Pre-2020 Score | Post-2020 Score | Change |
|---------|---------------|-----------------|--------|
| Outokumpu | 73.6 | 92.5 | **+18.9** |
| Salzgitter AG | 19.7 | 34.6 | **+14.9** |
| SHS Group | 14.6 | 28.6 | **+14.1** |
| SSAB | 28.9 | 40.1 | **+11.2** |
| Voestalpine | 14.1 | 22.0 | **+7.9** |
| ArcelorMittal | 28.4 | 32.0 | **+3.6** |
| Feralpi Group | 69.1 | 69.1 | ≈0 (already strong) |

Companies with data only post-2020 (insufficient pre-COVID history): Acciaierie d'Italia, Acerinox EU, Celsa, SIDENOR, Tata Steel Nederland, Tata Steel UK.

Notable findings:
- **Outokumpu** is the only company to improve its trend score meaningfully post-2020 (23.1 → 30.0), reflecting a genuine emission reduction trajectory
- **BF-BOF companies' score increases** are driven almost entirely by technology plan announcements (tech score component), not by achieved emission reductions — their trend scores remain at 0.0 across both periods
- **Voestalpine pre-2020 trend** showed increasing emissions (slope +6.6%/yr, R²=0.97) — the direction reversed post-2020 but still doesn't reach the −2%/yr threshold

→ `results/models/action_scores_comparison.csv`
→ `results/models/action_scores_temporal.csv` (full per-company, per-period breakdown)
→ `results/models/talk_vs_action.mp4` (animated Talk Score vs Action Score matrix, 2019→2024)

---

## 4. NLP — What Companies Say

**Source:** `03_nlp/run_all.ipynb`
**Corpus:** 197 sustainability reports, 15 EU steel companies, 2013–2025

### ClimateBERT Analysis

Five ClimateBERT classifiers were run across all report chunks:

| Score | Meaning |
|-------|---------|
| **Detector** | Is the chunk climate-related? |
| **Specificity** | Concrete/quantitative vs. vague language |
| **Commitment** | Does it commit to action? |
| **Sentiment** | Opportunity / neutral / risk framing |
| **Net-zero** | Net-zero or similar target language |

**Talk Score** = 20% climate volume + 40% specificity + 40% commitment (0–100 composite). Tracks quality of climate communication, not just volume.

Key trends visible in `results/bert/`:
- Talk scores trend upward across the corpus, especially post-2018
- Net-zero language spikes after the Paris Agreement and EU Green Deal
- Sentiment shifts toward opportunity framing in more recent reports

Charts: `talk_score_trend.png`, `talk_score_per_company.png`, `wordclouds.png`, `funnel_trend.png`
Data: `company_year.csv`, `yearly_industry.csv`, `funnel_company_year.csv`

### Barriers to Decarbonization (Topic Modeling)

RAG extraction identified decarbonization barriers from 197 reports; BERTopic clustered them into coherent themes. Top 6 (from `results/topics2/`):

| Rank | Topic | Documents |
|------|-------|-----------|
| 1 | **Limited Carbon Steel Decarbonisation** | 456 |
| 2 | **Carbon Certification & Emissions Costs** | 380 |
| 3 | **Integrating Waste Treatment Complexity** | 215 |
| 4 | **Industrial Waste to Hydrogen** | 65 |
| 5 | **Renewable Energy Availability Limitations** | 64 |
| 6 | **Industrial Carbon Technology Limitations** | 46 |

The dominant barriers are **market-side**: companies report they cannot procure enough low-carbon steel inputs and face high costs for carbon certification. Technical integration complexity (waste streams, biomass, hydrogen conversion) comes third. Direct technology constraints rank lower — consistent with the quantitative finding that technology type is already locked in; the barrier is accessing the clean alternative, not understanding the need.

Full topic set (17 topics): `results/topics1/barriers_labels.csv`

→ `results/topics2/barriers_documents.html` — interactive explorer
→ `results/topics2/barriers_datamap.png` — 2D cluster map
→ `results/topics2/barriers_barchart.html` — keyword barchart

### Motivators for Decarbonization (Topic Modeling)

Top 6 (from `results/topics2/`):

| Rank | Topic | Documents |
|------|-------|-----------|
| 1 | **Reducing Carbon Emissions** | 314 |
| 2 | **Sustainable Supply Chain Practices** | 291 |
| 3 | **Low Carbon Steel Technologies** | 252 |
| 4 | **Energy Efficiency Opportunities** | 55 |
| 5 | **Community Engagement Challenges** | 41 |
| 6 | **Employee Engagement & Satisfaction** | 24 |

The top motivators are **internally driven**: voluntary emissions reduction commitments and technology transition narratives dominate, ahead of regulatory compliance. The supply chain dimension (sustainable practices, circular economy, recycling) is the second largest cluster — suggesting customer and upstream pressure as a significant motivator beyond regulation.

Note: topics 1 and 2 in the motivators set reflect merges of several closely related themes from the original 16-topic model. Full pre-merge topic set: `results/topics1/motivators_labels.csv`.

→ `results/topics2/motivators_documents.html` — interactive explorer
→ `results/topics2/motivators_datamap.png` — 2D cluster map
→ `results/topics2/motivators_barchart.html` — keyword barchart

### Extraction Coverage

| Category | Companies | Reports |
|----------|-----------|---------|
| Barriers | 15 | 197 |
| Motivators | 15 | 197 |

Per-company CSVs with all extracted items: `results/rag/barriers_*.csv`, `results/rag/motivators_*.csv`

---

## 5. Data & Outputs

All pre-computed outputs are committed to `results/` and can be browsed without running the pipeline.

### `results/EDA/` — Emissions Exploratory Analysis

| File | Contents |
|------|----------|
| `01_boxplot_technology.png` | Emission intensity distribution by technology |
| `02_scope1_timeseries_european.png` | Company-level Scope 1 trends vs. EU/global benchmarks |
| `03_scope12_timeseries_european.png` | Scope 1+2 operational intensity trends |
| `04_production_vs_intensity.png` | Production scale vs. emission intensity |
| `05_scope2_by_company.png` | Scope 2 electricity intensity per company |
| `07_carbon_price_paradox.png` | EU ETS price vs. sector intensity (dual-axis) |
| `stats_by_technology.csv` | Descriptive statistics by technology group |

### `results/models/` — Baseline & Action Score

| File | Contents |
|------|----------|
| `baseline_tech_comparison.png` | Technology baseline model plot |
| `baseline_model_dataset.csv` | 103 obs with intensities, predictions, residuals |
| `action_score_breakdown.png` | Action score component breakdown per company |
| `action_scores_total.csv` | Full action score with all dimensions, 13 companies |
| `action_scores_comparison.csv` | Pre/post-COVID score comparison |
| `action_scores_temporal.csv` | Per-company, per-period full breakdown |
| `talk_vs_action.mp4` | Animated Talk Score vs Action Score matrix |

### `results/bert/` — ClimateBERT Analysis

| File | Contents |
|------|----------|
| `talk_score_trend.png` | Industry-wide talk score over time |
| `talk_score_per_company.png` | Per-company talk score trajectories |
| `wordclouds.png` | Net-zero language word clouds |
| `funnel_trend.png` | ClimateBERT scoring funnel over time |
| `company_year.csv` | BERT scores per company × year |
| `yearly_industry.csv` | Aggregated yearly industry scores |
| `funnel_company_year.csv` | Net-zero funnel data per company × year |

### `results/topics2/` — Topic Modeling Deliverable (top 6 per category)

| File | What it shows |
|------|--------------|
| `barriers_barchart.html` | Top keywords per barrier topic |
| `barriers_documents.html` | Explore barrier extractions interactively |
| `barriers_topics_2d.html` | 2D topic space |
| `barriers_over_time.html` | Barrier topic frequency by year |
| `barriers_per_company.html` | Topic breakdown per company |
| `barriers_datamap.png` | 2D cluster map (static) |
| `motivators_barchart.html` | Top keywords per motivator topic |
| `motivators_documents.html` | Explore motivator extractions interactively |
| `motivators_topics_2d.html` | 2D topic space — motivators |
| `motivators_over_time.html` | Motivator topic frequency by year |
| `motivators_per_company.html` | Topic breakdown per company |
| `motivators_datamap.png` | 2D cluster map (static) |

**`results/topics1/`** — full topic set (17 barriers, 16 motivators, same file structure).
**`results/rag/`** — per-company barriers & motivators CSVs (15 companies).
**`results/final_presentation.pdf`** — final project presentation.

---

## Methodology Notes

- **Corpus:** 197 PDF sustainability reports, 15 EU steel companies, 2013–2025; translated to English where needed
- **Emissions data:** Company-reported Scope 1+2; EU ETS verified emissions (EEA); cross-validated against source reports; 13 companies with ≥4 years Scope 1 data used for analysis
- **BERT filtering:** ClimateBERT detector threshold 0.7 reduces NLP corpus ~10× before LLM extraction
- **RAG extraction:** FAISS-indexed chunks (Snowflake arctic-embed-s), top-20 retrieved per query (MMR diversity), Llama-3.1-8b-instant via Groq extracts bullet-point barriers/motivators
- **Topic modeling:** BERTopic with IBM Granite embedding (granite-embedding-english-r2), HDBSCAN (barriers: mcs=25, ms=5, eom; motivators: mcs=16, ms=3, eom), UMAP (barriers: 5/15; motivators: 5/25), c-TF-IDF keywords → LLM labels
- **Action score:** 100-pt composite (performance 30, trend 30, data quality 15, technology 20, renewable 5); trend threshold −2%/yr SBTi-aligned, R²≥0.5 required; assessed through 2024
- **Panel models:** Entity fixed-effects with Driscoll-Kraay standard errors; Granger causality (lag 2); DiD with parallel-trends assumption

---

*SuSteelAible — March 2026*
[@am0ebe](https://github.com/am0ebe) · [@calluna-borealis](https://github.com/calluna-borealis) · [@dzyen](https://github.com/dzyen) · [@aposkoub92](https://github.com/aposkoub92) · [@MJR-data](https://github.com/MJR-data)
