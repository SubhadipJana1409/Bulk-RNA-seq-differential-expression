"""
Simple pathway enrichment analysis using Fisher's exact test.

Gene sets are curated IBD-relevant KEGG/Reactome pathways.
For real datasets, swap in full MSigDB GMT files.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


# ── Curated IBD-relevant gene sets ───────────────────────────────────────────
# Sources: KEGG hsa05321 (IBD), Reactome R-HSA-913531, R-HSA-6785807
GENE_SETS: dict[str, list[str]] = {
    "Cytokine-cytokine receptor interaction (KEGG)": [
        "IL6", "IL1B", "TNF", "IL8", "CXCL1", "CXCL2", "CXCL3",
        "CXCL9", "CXCL10", "CXCL11", "IL17A", "IL17F", "IL22",
        "IFNG", "IL2", "IL4", "IL10", "IL12A", "IL18",
    ],
    "Interferon signaling (Reactome)": [
        "STAT1", "STAT2", "IRF1", "IRF7", "IRF9",
        "IFIT1", "IFIT2", "IFIT3", "ISG15", "MX1", "OAS1", "OAS2",
        "GBP1", "GBP2", "GBP4", "GBP5", "IDO1",
    ],
    "NOD-like receptor signaling (KEGG)": [
        "NOD2", "RIPK2", "CARD9", "NLRP3", "PYCARD",
        "IL1B", "IL18", "CASP1", "NLRC4",
    ],
    "Intestinal barrier & tight junctions": [
        "CLDN3", "CLDN8", "OCLN", "TJP1", "TJP2",
        "MUC2", "MUC5B", "TFF3", "SPDEF", "KLF4",
        "AQP8", "SLC26A3", "SLC26A2",
    ],
    "Reactive oxygen species & DUOX pathway": [
        "DUOX2", "DUOXA2", "NOX1", "CYBB", "NCF1", "NCF2", "NCF4",
        "SOD2", "GPX2",
    ],
    "Antigen presentation (MHC class II)": [
        "HLA-DRA", "HLA-DRB1", "HLA-DQB1", "HLA-DPA1", "HLA-DPB1",
        "CD74", "CIITA", "TAPBP", "B2M",
    ],
    "Neutrophil degranulation (Reactome)": [
        "S100A8", "S100A9", "S100A12", "FCGR3B", "FCGR2A",
        "MMP3", "MMP9", "MMP12", "ELANE", "MPO", "LCN2",
        "OLFM4",
    ],
    "Th17 cell differentiation (KEGG)": [
        "RORC", "IL17A", "IL17F", "IL22", "IL6", "TGFB1",
        "STAT3", "IRF4",
    ],
    "Carbohydrate absorption (KEGG)": [
        "SI", "FABP1", "FABP2", "CA1", "CA2", "CA4",
        "SLC5A1", "SLC2A2", "ALDOB", "PCK1",
    ],
    "Complement & coagulation cascade": [
        "C3", "C4A", "C4B", "CFB", "CFD", "CFH",
        "F2", "F3", "F7", "SERPINA1",
    ],
}


def run_enrichment(
    de_results: pd.DataFrame,
    gene_universe: Optional[list[str]] = None,
    padj_threshold: float = 0.05,
    lfc_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Fisher's exact test enrichment for significant DE genes against curated gene sets.

    Parameters
    ----------
    de_results    : DE results DataFrame with 'padj' and 'log2FoldChange' columns
    gene_universe : all tested genes (defaults to de_results index)
    padj_threshold: significance cutoff for DE gene selection
    lfc_threshold : |log2FC| cutoff for DE gene selection

    Returns
    -------
    DataFrame with columns: pathway, n_genes, n_overlap, odds_ratio, pvalue, padj, sig_genes
    """
    if gene_universe is None:
        gene_universe = de_results.index.tolist()

    n_universe = len(gene_universe)

    # Significant DE gene set
    sig_mask = (
        (de_results["padj"] < padj_threshold) &
        (de_results["log2FoldChange"].abs() >= lfc_threshold)
    )
    sig_genes = set(de_results.index[sig_mask])
    n_sig = len(sig_genes)

    rows = []
    for pathway, members in GENE_SETS.items():
        members_in_universe = [g for g in members if g in gene_universe]
        members_in_sig      = [g for g in members_in_universe if g in sig_genes]

        n_path     = len(members_in_universe)
        n_overlap  = len(members_in_sig)

        if n_path == 0:
            continue

        # 2×2 contingency table
        a = n_overlap                        # sig  & in pathway
        b = n_sig - n_overlap                # sig  & not in pathway
        c = n_path - n_overlap               # !sig & in pathway
        d = n_universe - n_sig - c           # !sig & not in pathway

        table = [[a, b], [c, d]]
        odds_ratio, pvalue = fisher_exact(table, alternative="greater")

        rows.append({
            "pathway":      pathway,
            "n_pathway_genes": n_path,
            "n_overlap":    n_overlap,
            "odds_ratio":   round(odds_ratio, 3),
            "pvalue":       pvalue,
            "sig_genes":    ", ".join(sorted(members_in_sig)),
        })

    enrich_df = pd.DataFrame(rows)

    if len(enrich_df) == 0:
        return enrich_df

    # BH correction
    _, padj, _, _ = multipletests(enrich_df["pvalue"], method="fdr_bh")
    enrich_df["padj"] = padj
    enrich_df = enrich_df.sort_values("pvalue")

    n_enriched = (enrich_df["padj"] < 0.05).sum()
    logger.info("Pathway enrichment: %d/%d pathways significant (padj<0.05)", n_enriched, len(enrich_df))

    return enrich_df
