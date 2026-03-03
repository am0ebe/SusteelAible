# SuSteelAible — Results

**Decarbonization in European Steel: What 200 Sustainability Reports Tell Us**

This document summarizes the key findings from the SuSteelAible project. The analysis combines quantitative emissions data (16 companies, 2013–2024) with NLP analysis of ~200 corporate sustainability reports to understand technology lock-in, policy effectiveness, and the barriers and motivators companies articulate around decarbonization.

> **Interactive outputs** (topic maps, document explorer, barchart visualizations) are HTML files in `results/topics/` — open them in a browser after cloning. The final presentation slide deck is at `results/final_presentation.pdf`.

---

## Table of Contents

1. [Key Findings at a Glance](#1-key-findings-at-a-glance)
2. [Emissions Analysis](#2-emissions-analysis)
3. [Modeling — What Drives Emissions?](#3-modeling--what-drives-emissions)
4. [NLP — What Companies Say](#4-nlp--what-companies-say)
5. [Interactive Visualizations](#5-interactive-visualizations)

---

## 1. Key Findings at a Glance

| Finding | Detail |
|---------|--------|
| **Technology explains 80% of emission variance** | BF-BOF plants emit ~1.8 tCO₂e/t; EAF plants ~0.6 tCO₂e/t — a 67% gap driven entirely by production method |
| **Carbon prices rose 17×, emissions stayed flat** | EU ETS prices climbed from ~€5 to €85+/tCO₂ (2013–2023); sector-wide intensity did not budge |
| **Policy interventions show no within-firm effect** | ETS, CBAM, and the Green Deal are not associated with statistically significant intensity reductions at firm level |
| **Coal prices Granger-cause intensity** | Coal price movements predict intensity changes with a ~2-year lag; carbon prices show borderline effects |
| **Scope 2 emissions are falling for EAF companies** | Grid decarbonization, not steel process improvements, is driving the only visible downward trend |
| **Companies primarily cite market and certification barriers** | Top barrier themes: carbon steel availability, certification/labeling costs, waste integration complexity |
| **Motivators are commitment-driven, not just regulatory** | Companies cite internal emissions targets and low-carbon steel transition as top motivators |
| **Climate discourse is intensifying** | "Talk scores" (specificity + commitment + volume) trend upward across the reporting corpus |

---

## 2. Emissions Analysis

**Source:** `01_eda/EDA_emissions.ipynb` and `01_eda/external_drivers_eda_and_model.ipynb`
**Data:** 13–16 European steel companies, 2013–2024, Scope 1+2 emissions

### The Technology Gap

The single largest determinant of emission intensity is production technology:

| Technology | Avg. Intensity (Scope 1) | Range |
|------------|--------------------------|-------|
| BF-BOF (blast furnace) | ~1.74 tCO₂e/t | 1.38–2.54 |
| EAF Stainless | ~0.44 tCO₂e/t | 0.35–0.53 |
| EAF (standard) | ~0.10 tCO₂e/t | 0.08–0.13 |

BF-BOF plants use iron ore and coking coal — a chemically constrained process that cannot be incrementally improved away from fossil fuels. EAF plants melt scrap steel with electricity, which can be decarbonized through grid improvements.

### The Carbon Price Paradox

EU ETS carbon prices increased 17-fold between 2013 and 2023 (from ~€5 to €85+/tCO₂). Over the same period, EU-wide steel emission intensity remained stable at ~0.7–0.8 tCO₂e/t.

This is not a market failure — it reflects **technology lock-in**. Existing BF-BOF plants cannot respond to price signals by switching production method. Decarbonization requires new capital investment in EAF or H₂-DRI capacity, not incremental efficiency gains.

### Scope 2 Progress

EAF companies show a visible downward trend in total operational intensity (Scope 1+2), driven primarily by grid decarbonization (lower carbon electricity), not by changes to steel production itself. BF-BOF Scope 1 emissions remain structurally flat.

### External Drivers

Panel fixed-effects regressions controlling for firm-level heterogeneity show:
- **Only firm age** has a statistically significant within-firm effect on intensity
- **ETS, CBAM, Green Deal** show no significant within-firm intensity reduction
- **Difference-in-differences** analysis: best treatment year 2019 (p=0.037) disappears with controls
- **Granger causality:** Coal prices Granger-cause intensity at lag 2 (p=0.0015); carbon prices are borderline (p=0.0018)

The implication: market signals (input costs) have more immediate causal traction on emission intensity than policy interventions, likely because they affect operating decisions within existing technology constraints.

---

## 3. Modeling — What Drives Emissions?

**Source:** `02_models/baseline_model.ipynb`, `02_models/predictions.ipynb`, `02_models/action_score_concept.ipynb`

### Technology as Baseline

A single-feature decision tree using technology type (EAF vs. BF-BOF) as the only predictor achieves:

- **R² = 0.798** — technology explains ~80% of emission intensity variance
- **MAE = 0.24 tCO₂e/t**
- **p < 0.001** (highly significant t-test)

This sets a hard ceiling: any additional factors (pricing, policy, corporate commitments) can explain at most the remaining ~20% of variance.

### Action Score Framework

A 100-point composite score was developed to assess decarbonization readiness per company:

| Dimension | Weight | Measure |
|-----------|--------|---------|
| Performance | 30 pts | Current intensity vs. 2.0 tCO₂e/t benchmark |
| Trend | 30 pts | Annual intensity improvement rate (threshold: −2%/yr, SBTi-aligned) |
| Data Quality | 15 pts | Reporting completeness and time-series length |
| Technology | 20 pts | Current tech and stated transition plans |
| Renewable | 5 pts | Renewable electricity procurement (EAF companies only) |

A pre/post-COVID comparison (`action_score_temporal.ipynb`) identifies which companies accelerated decarbonization efforts post-2020 and how the EU Green Deal and CBAM translated (or failed to translate) into measurable action.

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

**Talk Score** = 20% climate volume + 40% specificity + 40% commitment (0–100 composite). This tracks the quality of climate communication over time, not just quantity.

Key trends visible in `results/bert/`:
- Talk scores trend upward across the corpus, especially post-2018
- Net-zero language spikes after the Paris Agreement and EU Green Deal
- Sentiment shifts toward opportunity framing in more recent reports

Charts: `talk_score_trend.png`, `talk_score_per_company.png`, `wordclouds.png`, `funnel_trend.png`
Data: `company_year.csv`, `yearly_industry.csv`, `funnel_company_year.csv`

### Barriers to Decarbonization (Topic Modeling)

RAG extraction identified decarbonization barriers from 197 reports, then BERTopic clustered them into coherent themes. Top 6 (from `results/topics2/`):

| Rank | Topic | Documents |
|------|-------|-----------|
| 1 | **Limited Carbon Steel Decarbonisation** | 456 |
| 2 | **Carbon Certification & Emissions Costs** | 380 |
| 3 | **Integrating Waste Treatment Complexity** | 215 |
| 4 | **Industrial Waste to Hydrogen** | 65 |
| 5 | **Renewable Energy Availability Limitations** | 64 |
| 6 | **Industrial Carbon Technology Limitations** | 46 |

The dominant barriers are **market-side**: companies can't procure enough low-carbon steel inputs and face high costs for carbon certification. Technical integration complexity (waste streams, biomass) comes third. Direct technology constraints rank lower — consistent with the quantitative finding that the biggest driver is technology type, which is already locked in.

Full topic set (17 topics): `results/topics1/barriers_labels.csv`

→ Interactive explorer: `results/topics2/barriers_documents.html`
→ Cluster map: `results/topics2/barriers_datamap.png`
→ Barchart: `results/topics2/barriers_barchart.html`

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

The top motivators are **internally driven**: voluntary emissions targets and technology transition narratives dominate, ahead of regulatory compliance. This suggests that the companies most active in reporting are already committed beyond the regulatory minimum — though that may reflect selection bias in who publishes detailed sustainability reports.

Full topic set (16 topics): `results/topics1/motivators_labels.csv`

→ Interactive explorer: `results/topics2/motivators_documents.html`
→ Cluster map: `results/topics2/motivators_datamap.png`
→ Barchart: `results/topics2/motivators_barchart.html`

### Extraction Coverage

| Category | Companies | Reports |
|----------|-----------|---------|
| Barriers | 15 | 197 |
| Motivators | 15 | 197 |

Per-company CSVs with all extracted items: `results/rag/barriers_*.csv`, `results/rag/motivators_*.csv`

---

## 5. Interactive Visualizations

Download/clone the repo and open HTML files in a browser.

**`results/topics2/`** — merged deliverable (top 6 topics per category):

| File | What it shows |
|------|--------------|
| `barriers_barchart.html` | Top keywords per barrier topic |
| `barriers_documents.html` | Explore individual barrier extractions by topic |
| `barriers_topics_2d.html` | 2D topic space |
| `barriers_over_time.html` | Barrier topic frequency by year |
| `barriers_per_company.html` | Topic breakdown per company |
| `barriers_datamap.png` | 2D cluster map (static) |
| `motivators_barchart.html` | Top keywords per motivator topic |
| `motivators_documents.html` | Explore individual motivator extractions by topic |
| `motivators_topics_2d.html` | 2D topic space — motivators |
| `motivators_over_time.html` | Motivator topic frequency by year |
| `motivators_per_company.html` | Topic breakdown per company |
| `motivators_datamap.png` | 2D cluster map (static) |

**`results/topics1/`** — full topic set (17 barriers, 16 motivators, same file names).

**Final presentation:** `results/final_presentation.pdf`

---

## Methodology Notes

- **Corpus:** 197 PDF sustainability reports, 15 EU steel companies, 2013–2025
- **Preprocessing:** PDF → text chunks (600–1600 chars), translated to English where needed
- **BERT filtering:** ClimateBERT detector (threshold 0.7) reduces corpus ~10× before LLM extraction
- **RAG extraction:** FAISS-indexed chunks (Snowflake arctic-embed-s), top-20 retrieved per query, LLM (Llama-3.1-8b-instant via Groq) extracts bullet-point barriers/motivators
- **Topic modeling:** BERTopic with IBM Granite embedding (granite-embedding-english-r2), HDBSCAN clustering, c-TF-IDF keywords → LLM labels
- **Emissions data:** Company-reported Scope 1+2 data, EU ETS verified emissions (EEA), cross-validated against source reports

---

*SuSteelAible — March 2026*
[@am0ebe](https://github.com/am0ebe) · [@calluna-borealis](https://github.com/calluna-borealis) · [@dzyen](https://github.com/dzyen) · [@aposkoub92](https://github.com/aposkoub92) · [@MJR-data](https://github.com/MJR-data)
