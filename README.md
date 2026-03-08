# Day 21 · Bulk RNA-seq Differential Expression

**IBD vs Control · PyDESeq2 · 9 publication-quality figures**

Part of a 30-day bioinformatics challenge.
Previous: [Day 20 – scRNA-seq Gut Clustering](https://github.com/SubhadipJana1409/Bulk-RNA-seq-differential-expression)

---

## Overview

This project implements a complete **bulk RNA-seq differential expression (DE) analysis** pipeline comparing Inflammatory Bowel Disease (IBD) mucosal biopsies to healthy controls. It uses [PyDESeq2](https://github.com/owkin/PyDESeq2) — a faithful Python port of the canonical R package DESeq2 (Love et al. 2014) — for statistically rigorous DE testing via negative-binomial GLMs and Wald tests.

Key biological motivation: IBD involves disrupted intestinal homeostasis, dysregulated immune responses, and epithelial barrier dysfunction. DE analysis on bulk RNA-seq from mucosal biopsies identifies the molecular drivers of this dysregulation.

---

## Biological Background

The pipeline models key features of published IBD transcriptomics datasets (Corridoni et al. 2020; Vanhove et al. 2023):

| Category | Genes | Direction in IBD |
|---|---|---|
| Interferon response | IFIT1, IFIT2, IFIT3, ISG15, MX1, OAS1/2 | ↑ Up |
| Cytokine signalling | IL6, IL1B, TNF, CXCL9, CXCL10, CXCL11 | ↑ Up |
| Neutrophil/S100 proteins | S100A8, S100A9, S100A12 | ↑ Up |
| Th17 differentiation | IL17A, IL17F, RORC | ↑ Up |
| Epithelial barrier | MUC2, TFF3, CLDN3, CLDN8 | ↓ Down |
| Ion transport | SLC26A3, AQP8, CA1, CA2 | ↓ Down |
| Absorptive function | FABP1, SI, ALDOB | ↓ Down |

---

## Pipeline Architecture

```
Count Matrix (genes × samples)
         │
         ▼
┌─────────────────────┐
│  1. Low-count filter│  Remove genes < 10 counts in < 4 samples
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  2. DESeq2 Model    │  Size factors → NB dispersion → GLM → Wald test
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  3. Wald Test       │  IBD vs Control contrast; Benjamini-Hochberg FDR
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  4. Pathway Enrich  │  Fisher's exact test on 10 curated gene sets
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  5. 9 Figures       │  QC, PCA, distances, MA, volcano, heatmap, ...
└─────────────────────┘
```

---

## Figures

| Figure | Description |
|--------|-------------|
| `fig1_sample_qc.png`          | Library size & detected genes per sample |
| `fig2_pca.png`                | PCA of VST-normalised counts (PC1/2, PC1/3) |
| `fig3_sample_distance.png`    | Euclidean sample-distance clustered heatmap |
| `fig4_ma_plot.png`            | Mean expression vs log₂FC (MA plot) |
| `fig5_volcano.png`            | −log₁₀(padj) vs log₂FC volcano |
| `fig6_heatmap_deg.png`        | Z-score heatmap of top 50 DEGs |
| `fig7_dispersion.png`         | Mean-dispersion relationship |
| `fig8_pathway_enrichment.png` | Fisher's exact pathway enrichment barplot |
| `fig9_de_summary.png`         | Summary: gene counts, padj sensitivity, |LFC| distribution |

---

## Output Files

```
outputs/
├── de_results_all.csv           # All tested genes with stats
├── de_results_significant.csv   # Significant DEGs only (padj<0.05, |lfc|≥1)
├── normalized_counts.csv        # Size-factor normalised counts
├── vst_counts.csv               # VST-transformed counts (for PCA / heatmap)
├── sample_metadata.csv          # Sample annotations
├── pathway_enrichment.csv       # Enrichment results
└── fig1_*.png … fig9_*.png      # All figures
```

---

## Quick Start

```bash
git clone https://github.com/SubhadipJana1409/day21-bulk-rnaseq-de
cd day21-bulk-rnaseq-de
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run full pipeline (simulated IBD data)
python -m src.main
```

### Using Real Data

Edit `configs/config.yaml`:

```yaml
data:
  counts_path:   "path/to/raw_counts.csv"   # genes × samples
  metadata_path: "path/to/metadata.csv"     # must have 'condition' column
```

The metadata file must include a `condition` column with exactly two unique values (e.g. `IBD`, `Control`). The reference level is set in `configs/config.yaml`.

---

## Configuration

```yaml
data:
  n_samples:        24     # (simulation) total samples
  n_genes:        8000     # (simulation) total genes
  effect_size:       2.5   # (simulation) log2FC for DE genes

de:
  reference_level:  "Control"
  padj_threshold:    0.05
  lfc_threshold:     1.0   # |log2FC| ≥ 1 → 2-fold change
  min_count:        10
  min_samples:       4
```

---

## Project Structure

```
day21-bulk-rnaseq-de/
├── src/
│   ├── data/
│   │   └── simulator.py        # NB count simulator (IBD vs Control)
│   ├── models/
│   │   ├── deseq2.py           # PyDESeq2 wrapper
│   │   └── enrichment.py       # Fisher's exact pathway enrichment
│   ├── visualization/
│   │   └── plots.py            # All 9 figures
│   ├── utils/
│   │   ├── config.py
│   │   └── logger.py
│   └── main.py
├── tests/
│   ├── test_simulator.py       # 9 tests
│   ├── test_deseq2.py          # 14 tests
│   └── test_plots_enrichment.py # 14 tests
├── configs/config.yaml
├── outputs/                    # Generated results & figures
├── notebooks/
│   └── 01_bulk_rnaseq_exploration.ipynb
└── requirements.txt
```

---

## Tests

```bash
pytest tests/ -v
# 37 passed
```

---

## Methods

**Differential Expression**: PyDESeq2 v0.5+ (Muzellec et al. 2023), which implements the DESeq2 method (Love et al. 2014). The pipeline estimates size factors by the median-of-ratios method, fits gene-wise negative-binomial dispersions with empirical Bayes shrinkage, fits a GLM with design matrix `~ condition`, and uses the Wald test for the IBD vs Control contrast. P-values are adjusted using the Benjamini-Hochberg procedure.

**Significance threshold**: padj < 0.05, |log₂FC| ≥ 1 (2-fold change).

**Pathway enrichment**: Fisher's exact test (one-sided, "greater") on 10 curated IBD-relevant gene sets (KEGG, Reactome). Multiple testing correction: Benjamini-Hochberg.

---

## References

1. Love MI, Huber W, Anders S (2014). Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. *Genome Biology*, 15, 550.
2. Muzellec B, Telezhkin A, Alam R et al. (2023). PyDESeq2: a python package for bulk RNA-seq differential expression analysis. *Bioinformatics*.
3. Corridoni D et al. (2020). Single-cell atlas of colonic CD8+ T cells in ulcerative colitis. *Nature Medicine*, 26, 1480–1490.
4. Vanhove W et al. (2023). Transcriptome analysis of intestinal biopsies from Crohn's disease patients. *Journal of Crohn's and Colitis*.

---

## License

MIT
