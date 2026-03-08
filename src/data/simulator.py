"""
Simulate realistic bulk RNA-seq count data for IBD vs Control.

Uses a negative-binomial model with gene-wise dispersion and condition-specific
means, mimicking published IBD datasets (Vanhove et al. 2023; Corridoni et al. 2020).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ── Published IBD-associated genes (Corridoni 2020, Smillie 2019) ──────────
IBD_UP_GENES = [
    "CXCL10", "CXCL9", "CXCL11", "IDO1", "GBP1", "GBP2", "GBP4", "GBP5",
    "IFIT1", "IFIT2", "IFIT3", "ISG15", "MX1", "OAS1", "OAS2",
    "STAT1", "STAT2", "IRF1", "IRF7", "IRF9",
    "IL6", "IL1B", "TNF", "IL8", "CXCL1", "CXCL2", "CXCL3",
    "S100A8", "S100A9", "S100A12",
    "FCGR3B", "FCGR2A", "CD68", "MMP3", "MMP9", "MMP12",
    "IL17A", "IL17F", "IL22",
    "RORC", "TBX21",
    "OLFM4", "REG1A", "REG1B", "REG3A",
    "DUOX2", "DUOXA2",
    "HLA-DRA", "HLA-DRB1", "HLA-DQB1",
]

IBD_DOWN_GENES = [
    "CA1", "CA2", "CA4",
    "SLC26A3", "SLC26A2",
    "AQP8", "CLCA1", "SPDEF",
    "MUC2", "MUC5B", "TFF3",
    "CEACAM7", "CEACAM1",
    "SI", "FABP1", "FABP2",
    "GUCA2A", "GUCA2B",
    "KLF4", "CDX2",
    "HEPACAM2", "ANPEP",
    "LGALS4", "LGALS9",
    "ADH1C", "ALDOB",
    "PCK1", "APOC3",
    "ABCG5", "ABCG8",
    "DMBT1",
    "CLDN8", "CLDN3",
]

HOUSEKEEPING = [
    "ACTB", "GAPDH", "RPL13A", "RPL27", "RPS18", "RPLP0",
    "POLR2A", "EEF1A1", "HPRT1", "SDHA", "YWHAZ", "B2M",
    "UBC", "LDHA", "PPIA",
]


def simulate_counts(
    n_samples: int = 24,
    n_genes: int = 8000,
    n_ibd: Optional[int] = None,
    effect_size: float = 2.5,
    dispersion_mean: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a realistic count matrix for IBD vs Control.

    Parameters
    ----------
    n_samples  : total number of samples
    n_genes    : total number of genes
    n_ibd      : number of IBD samples (default: half of n_samples)
    effect_size: fold-change for DE genes (log2 scale)
    dispersion_mean: mean dispersion for NB model
    seed       : random seed

    Returns
    -------
    counts     : DataFrame (genes × samples)
    metadata   : DataFrame (samples × covariates)
    """
    rng = np.random.default_rng(seed)

    if n_ibd is None:
        n_ibd = n_samples // 2
    n_ctrl = n_samples - n_ibd

    # ── Sample metadata ──────────────────────────────────────────────────────
    ctrl_ids = [f"CTRL_{i+1:02d}" for i in range(n_ctrl)]
    ibd_ids  = [f"IBD_{i+1:02d}"  for i in range(n_ibd)]
    sample_ids = ctrl_ids + ibd_ids

    metadata = pd.DataFrame({
        "sample_id":  sample_ids,
        "condition":  ["Control"] * n_ctrl + ["IBD"] * n_ibd,
        "batch":      rng.choice(["A", "B"], size=n_samples).tolist(),
        "age":        rng.integers(20, 70, size=n_samples).tolist(),
        "sex":        rng.choice(["M", "F"], size=n_samples).tolist(),
        "rin_score":  np.round(rng.uniform(7.0, 9.5, size=n_samples), 2).tolist(),
    }).set_index("sample_id")

    # ── Gene pool ────────────────────────────────────────────────────────────
    special = IBD_UP_GENES + IBD_DOWN_GENES + HOUSEKEEPING
    n_background = max(0, n_genes - len(special))
    background_genes = [f"GENE_{i+1:05d}" for i in range(n_background)]
    gene_names = special + background_genes
    n_genes_actual = len(gene_names)

    # Gene-wise baseline expression (log-normal, mimicking real RNA-seq)
    log_mu_base = rng.normal(5.5, 2.2, size=n_genes_actual)
    mu_base = np.exp(log_mu_base).clip(10, 50_000)

    # Gene-wise dispersion (inverse-gamma-like, real: 0.01-2.0)
    dispersions = rng.gamma(shape=2.0, scale=dispersion_mean / 2.0, size=n_genes_actual).clip(0.01, 2.0)

    # Library-size factors (realistic: 10M–40M reads)
    lib_factors = rng.uniform(0.6, 1.6, size=n_samples)

    # ── Build count matrix ───────────────────────────────────────────────────
    counts = np.zeros((n_genes_actual, n_samples), dtype=np.int32)

    for s_idx in range(n_samples):
        is_ibd = s_idx >= n_ctrl
        mu_s = mu_base * lib_factors[s_idx]

        # Apply condition-specific fold changes
        for g_idx, gene in enumerate(gene_names):
            mu_g = mu_s[g_idx]
            if is_ibd:
                if gene in IBD_UP_GENES:
                    fc = rng.uniform(effect_size * 0.5, effect_size * 1.5)
                    mu_g = mu_g * (2 ** fc)
                elif gene in IBD_DOWN_GENES:
                    fc = rng.uniform(effect_size * 0.5, effect_size * 1.2)
                    mu_g = mu_g / (2 ** fc)

            # NB parameterisation: p = 1/(1+mu*disp), r = 1/disp
            disp = dispersions[g_idx]
            r = 1.0 / disp
            p = mu_g / (mu_g + r)
            counts[g_idx, s_idx] = rng.negative_binomial(r, 1 - p)

    counts_df = pd.DataFrame(
        counts,
        index=gene_names,
        columns=sample_ids,
    )

    # Ensure non-negative
    counts_df = counts_df.clip(lower=0)

    return counts_df, metadata


def load_or_simulate(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load from file if provided, otherwise simulate."""
    counts_path = cfg.get("counts_path")
    meta_path   = cfg.get("metadata_path")

    if counts_path and meta_path:
        counts   = pd.read_csv(counts_path, index_col=0)
        metadata = pd.read_csv(meta_path,   index_col=0)
        return counts, metadata

    return simulate_counts(
        n_samples=cfg.get("n_samples", 24),
        n_genes=cfg.get("n_genes", 8000),
        effect_size=cfg.get("effect_size", 2.5),
        dispersion_mean=cfg.get("dispersion_mean", 0.15),
        seed=cfg.get("seed", 42),
    )
