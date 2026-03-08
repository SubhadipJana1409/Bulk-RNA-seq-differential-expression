"""
Differential expression analysis using PyDESeq2.

PyDESeq2 is a faithful Python re-implementation of DESeq2 (Love et al. 2014),
using the same negative-binomial GLM, shrinkage estimators, and Wald test.

Reference: Muzellec et al. (2023) Bioinformatics.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from scipy.stats import false_discovery_control

logger = logging.getLogger(__name__)

PADJ_THRESHOLD  = 0.05
LFC_THRESHOLD   = 1.0   # |log2FC| ≥ 1  →  2-fold change


class DEAnalysis:
    """
    Wrapper around PyDESeq2 for a two-condition bulk RNA-seq DE analysis.

    Parameters
    ----------
    counts   : genes × samples count matrix (raw integers)
    metadata : samples × covariates DataFrame; must contain 'condition' column
    cfg      : configuration dictionary
    """

    def __init__(
        self,
        counts: pd.DataFrame,
        metadata: pd.DataFrame,
        cfg: Optional[dict] = None,
    ):
        self.counts   = counts
        self.metadata = metadata
        self.cfg      = cfg or {}

        self.dds: Optional[DeseqDataSet] = None
        self.results: Optional[pd.DataFrame] = None
        self.normalized_counts: Optional[pd.DataFrame] = None
        self.vst_counts: Optional[pd.DataFrame] = None

    # ── Pre-processing ────────────────────────────────────────────────────────

    def filter_low_counts(self, min_count: int = 10, min_samples: int = 4) -> pd.DataFrame:
        """
        Remove genes with fewer than `min_count` counts in at least `min_samples`.
        """
        mask = (self.counts >= min_count).sum(axis=1) >= min_samples
        filtered = self.counts.loc[mask]
        logger.info("Low-count filter: %d → %d genes", len(self.counts), len(filtered))
        return filtered

    # ── PyDESeq2 workflow ─────────────────────────────────────────────────────

    def run(
        self,
        reference_level: str = "Control",
        n_cpus: int = 1,
    ) -> pd.DataFrame:
        """
        Full DESeq2 pipeline:
          1. Filter low-count genes
          2. Fit DESeq2 model (size factors + dispersion)
          3. Wald test
          4. Annotate results (significant, direction)

        Returns
        -------
        results DataFrame with columns:
            baseMean, log2FoldChange, lfcSE, stat, pvalue, padj,
            significant, direction
        """
        logger.info("Starting DESeq2 pipeline …")

        # 1. Filter
        counts_filt = self.filter_low_counts(
            min_count=self.cfg.get("min_count", 10),
            min_samples=self.cfg.get("min_samples", 4),
        )

        # PyDESeq2 expects samples × genes
        counts_T = counts_filt.T.copy()
        # Ensure integer dtype
        counts_T = counts_T.astype(int)

        # Align metadata
        meta_aligned = self.metadata.loc[counts_T.index].copy()
        # Ensure condition is string (categorical)
        meta_aligned["condition"] = meta_aligned["condition"].astype(str)

        # 2. Build DeseqDataSet
        logger.info("Building DeseqDataSet …")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.dds = DeseqDataSet(
                counts=counts_T,
                metadata=meta_aligned,
                design_factors="condition",
                ref_level=["condition", reference_level],
                n_cpus=n_cpus,
                quiet=True,
            )
            self.dds.deseq2()

        # 3. Wald test
        logger.info("Running Wald test …")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Infer the non-reference condition name
            conditions = meta_aligned["condition"].unique().tolist()
            test_cond = [c for c in conditions if c != reference_level][0]
            contrast = ["condition", test_cond, reference_level]
            ds = DeseqStats(self.dds, contrast=contrast, quiet=True, n_cpus=n_cpus)
            ds.summary()

        self.results = ds.results_df.copy()

        # 4. Annotate
        padj_thr = self.cfg.get("padj_threshold", PADJ_THRESHOLD)
        lfc_thr  = self.cfg.get("lfc_threshold",  LFC_THRESHOLD)

        self.results["significant"] = (
            (self.results["padj"] < padj_thr) &
            (self.results["log2FoldChange"].abs() >= lfc_thr)
        )
        self.results["direction"] = np.where(
            self.results["log2FoldChange"] > 0, "Up", "Down"
        )
        self.results.loc[~self.results["significant"], "direction"] = "NS"

        # Normalized counts (DESeq2 size-factor normalised)
        sf = self.dds.obs["size_factors"].values  # per-sample (in obs)
        raw = counts_T.values.astype(float)
        norm = raw / sf[:, None]
        self.normalized_counts = pd.DataFrame(
            norm.T,
            index=counts_filt.index,
            columns=counts_T.index,
        )

        # Variance-stabilising transform (log2 normalised + 1 pseudocount)
        self.vst_counts = np.log2(self.normalized_counts + 1)

        n_sig = self.results["significant"].sum()
        n_up  = (self.results["direction"] == "Up").sum()
        n_dn  = (self.results["direction"] == "Down").sum()
        logger.info(
            "DE complete: %d significant (%d up, %d down) at padj<%.2f |lfc|≥%.1f",
            n_sig, n_up, n_dn, padj_thr, lfc_thr,
        )

        return self.results

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_sig_genes(self, direction: Optional[str] = None) -> pd.DataFrame:
        """Return significant DE genes, optionally filtered by direction."""
        if self.results is None:
            raise RuntimeError("Call run() first.")
        sig = self.results[self.results["significant"]]
        if direction in ("Up", "Down"):
            sig = sig[sig["direction"] == direction]
        return sig.sort_values("padj")

    def get_top_genes(self, n: int = 50) -> pd.DataFrame:
        """Top-n significant genes by adjusted p-value."""
        return self.get_sig_genes().head(n)

    def save_results(self, out_dir: str | Path) -> None:
        """Save results tables to CSV."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        if self.results is not None:
            self.results.to_csv(out / "de_results_all.csv")
            self.get_sig_genes().to_csv(out / "de_results_significant.csv")
            logger.info("Saved DE results to %s", out)

        if self.normalized_counts is not None:
            self.normalized_counts.to_csv(out / "normalized_counts.csv")

        if self.vst_counts is not None:
            self.vst_counts.to_csv(out / "vst_counts.csv")
