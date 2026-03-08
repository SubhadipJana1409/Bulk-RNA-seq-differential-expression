"""
Publication-quality visualizations for bulk RNA-seq differential expression.

Figures
-------
fig1  : Sample QC   – library size + detected genes per sample
fig2  : PCA         – PC1/PC2 of VST counts, coloured by condition
fig3  : Sample distance heatmap
fig4  : MA plot     – mean expression vs log2FC
fig5  : Volcano plot
fig6  : DEG heatmap – top 50 significant genes
fig7  : Dispersion  – mean-dispersion relationship
fig8  : Pathway enrichment barplot
fig9  : DE summary  – counts by direction + condition breakdown
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = {"IBD": "#E74C3C", "Control": "#3498DB"}
PALETTE_DIR = {"Up": "#E74C3C", "Down": "#3498DB", "NS": "#BDC3C7"}
FIGSIZE_STD = (10, 6)
FIGSIZE_SQ  = (8, 8)
DPI = 150


def _save(fig: plt.Figure, path: Path, name: str) -> None:
    out = path / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


# ── Figure 1: Sample QC ──────────────────────────────────────────────────────

def fig1_sample_qc(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Library size and detected-gene QC panels."""
    lib_size   = counts.sum(axis=0) / 1e6      # millions
    detected   = (counts > 0).sum(axis=0)

    conditions = metadata.loc[counts.columns, "condition"]
    colors     = [PALETTE.get(c, "gray") for c in conditions]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Sample Quality Control", fontsize=14, fontweight="bold", y=1.01)

    # Library size
    ax = axes[0]
    bars = ax.bar(range(len(lib_size)), lib_size.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(lib_size)))
    ax.set_xticklabels(lib_size.index, rotation=90, fontsize=6)
    ax.set_ylabel("Library Size (millions)")
    ax.set_title("Library Size per Sample")
    ax.axhline(lib_size.mean(), color="black", lw=1.5, ls="--", label=f"Mean: {lib_size.mean():.1f}M")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Detected genes
    ax = axes[1]
    ax.bar(range(len(detected)), detected.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(detected)))
    ax.set_xticklabels(detected.index, rotation=90, fontsize=6)
    ax.set_ylabel("Detected Genes (count > 0)")
    ax.set_title("Detected Genes per Sample")
    ax.axhline(detected.mean(), color="black", lw=1.5, ls="--", label=f"Mean: {int(detected.mean()):,}")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend
    patches = [mpatches.Patch(color=c, label=k) for k, c in PALETTE.items()]
    fig.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.08, 1.0), frameon=False)

    plt.tight_layout()
    _save(fig, out_dir, "fig1_sample_qc.png")


# ── Figure 2: PCA ─────────────────────────────────────────────────────────────

def fig2_pca(
    vst_counts: pd.DataFrame,
    metadata: pd.DataFrame,
    out_dir: Path,
) -> None:
    """PCA of VST-normalised counts."""
    X = vst_counts.T.values  # samples × genes
    pca = PCA(n_components=min(4, X.shape[0] - 1, X.shape[1]))
    coords = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_ * 100

    conditions = metadata.loc[vst_counts.columns, "condition"].values
    colors = [PALETTE.get(c, "gray") for c in conditions]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("PCA of VST-Normalised Counts", fontsize=14, fontweight="bold")

    for ax, (pc_x, pc_y) in zip(axes, [(0, 1), (0, 2)]):
        for cond, col in PALETTE.items():
            idx = np.where(conditions == cond)[0]
            ax.scatter(
                coords[idx, pc_x], coords[idx, pc_y],
                c=col, label=cond, s=80, alpha=0.85,
                edgecolors="white", linewidths=0.5,
            )
        ax.set_xlabel(f"PC{pc_x+1} ({var_exp[pc_x]:.1f}% variance)", fontsize=10)
        ax.set_ylabel(f"PC{pc_y+1} ({var_exp[pc_y]:.1f}% variance)", fontsize=10)
        ax.set_title(f"PC{pc_x+1} vs PC{pc_y+1}")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(fontsize=10, frameon=False)
    plt.tight_layout()
    _save(fig, out_dir, "fig2_pca.png")


# ── Figure 3: Sample distance heatmap ─────────────────────────────────────────

def fig3_sample_distance(
    vst_counts: pd.DataFrame,
    metadata: pd.DataFrame,
    out_dir: Path,
) -> None:
    X = vst_counts.T.values
    dist = squareform(pdist(X, metric="euclidean"))
    dist_df = pd.DataFrame(dist, index=vst_counts.columns, columns=vst_counts.columns)

    # Row/column colour annotation
    conditions = metadata.loc[vst_counts.columns, "condition"]
    row_colors = pd.Series(
        [PALETTE.get(c, "gray") for c in conditions],
        index=vst_counts.columns,
        name="Condition",
    )

    g = sns.clustermap(
        dist_df,
        cmap="Blues_r",
        row_colors=row_colors,
        col_colors=row_colors,
        figsize=(9, 8),
        xticklabels=True,
        yticklabels=True,
        linewidths=0.0,
    )
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=6, rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=6)
    g.figure.suptitle("Sample-to-Sample Euclidean Distance", y=1.01, fontsize=13, fontweight="bold")

    # Legend
    patches = [mpatches.Patch(color=c, label=k) for k, c in PALETTE.items()]
    g.ax_col_dendrogram.legend(handles=patches, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    g.figure.savefig(out_dir / "fig3_sample_distance.png", dpi=DPI, bbox_inches="tight")
    plt.close(g.figure)
    logger.info("Saved fig3_sample_distance.png")


# ── Figure 4: MA plot ─────────────────────────────────────────────────────────

def fig4_ma_plot(
    results: pd.DataFrame,
    out_dir: Path,
    padj_thr: float = 0.05,
    lfc_thr: float = 1.0,
) -> None:
    df = results.dropna(subset=["padj", "log2FoldChange", "baseMean"]).copy()
    df["log2baseMean"] = np.log2(df["baseMean"] + 1)
    df["color"] = df["significant"].map({True: None, False: PALETTE_DIR["NS"]})
    df.loc[df["significant"] & (df["direction"] == "Up"),   "color"] = PALETTE_DIR["Up"]
    df.loc[df["significant"] & (df["direction"] == "Down"), "color"] = PALETTE_DIR["Down"]

    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    for cat, col, label, zord in [
        ("NS",   PALETTE_DIR["NS"],   "Not significant", 1),
        ("Up",   PALETTE_DIR["Up"],   "Up-regulated",    3),
        ("Down", PALETTE_DIR["Down"], "Down-regulated",  2),
    ]:
        sub = df[df["direction"] == cat]
        ax.scatter(sub["log2baseMean"], sub["log2FoldChange"],
                   c=col, s=8, alpha=0.5, label=f"{label} (n={len(sub):,})", zorder=zord)

    ax.axhline(0,       color="black",  lw=1.0)
    ax.axhline( lfc_thr, color="gray",  lw=0.8, ls="--", alpha=0.7)
    ax.axhline(-lfc_thr, color="gray",  lw=0.8, ls="--", alpha=0.7)

    # Label top 8 genes
    top = df[df["significant"]].nsmallest(8, "padj")
    for _, row in top.iterrows():
        ax.annotate(
            row.name,
            (row["log2baseMean"], row["log2FoldChange"]),
            fontsize=7, xytext=(5, 0), textcoords="offset points",
        )

    ax.set_xlabel("log₂(Mean Normalised Count + 1)", fontsize=11)
    ax.set_ylabel("log₂ Fold Change (IBD / Control)", fontsize=11)
    ax.set_title("MA Plot", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, markerscale=2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir, "fig4_ma_plot.png")


# ── Figure 5: Volcano plot ────────────────────────────────────────────────────

def fig5_volcano(
    results: pd.DataFrame,
    out_dir: Path,
    padj_thr: float = 0.05,
    lfc_thr: float = 1.0,
) -> None:
    df = results.dropna(subset=["padj", "log2FoldChange"]).copy()
    df["neg_log10_padj"] = -np.log10(df["padj"].clip(lower=1e-300))

    fig, ax = plt.subplots(figsize=FIGSIZE_STD)

    for cat, col, label in [
        ("NS",   PALETTE_DIR["NS"],   "Not significant"),
        ("Up",   PALETTE_DIR["Up"],   "Up-regulated (IBD)"),
        ("Down", PALETTE_DIR["Down"], "Down-regulated (IBD)"),
    ]:
        sub = df[df["direction"] == cat]
        ax.scatter(sub["log2FoldChange"], sub["neg_log10_padj"],
                   c=col, s=10, alpha=0.55, label=f"{label} (n={len(sub):,})")

    ax.axvline( lfc_thr,          color="gray", lw=0.8, ls="--")
    ax.axvline(-lfc_thr,          color="gray", lw=0.8, ls="--")
    ax.axhline(-np.log10(padj_thr), color="gray", lw=0.8, ls="--")

    # Label top 12 genes
    top = df[df["significant"]].nsmallest(12, "padj")
    for _, row in top.iterrows():
        ax.annotate(
            row.name,
            (row["log2FoldChange"], row["neg_log10_padj"]),
            fontsize=7, xytext=(4, 2), textcoords="offset points",
        )

    ax.set_xlabel("log₂ Fold Change (IBD / Control)", fontsize=11)
    ax.set_ylabel("-log₁₀(adjusted p-value)", fontsize=11)
    ax.set_title("Volcano Plot: IBD vs Control", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, markerscale=2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir, "fig5_volcano.png")


# ── Figure 6: DEG heatmap ─────────────────────────────────────────────────────

def fig6_heatmap(
    vst_counts: pd.DataFrame,
    results: pd.DataFrame,
    metadata: pd.DataFrame,
    out_dir: Path,
    n_genes: int = 50,
) -> None:
    sig = results[results["significant"]].nsmallest(n_genes, "padj")
    if len(sig) == 0:
        logger.warning("No significant genes for heatmap")
        return

    # Subset and z-score per gene
    mat = vst_counts.loc[vst_counts.index.isin(sig.index)].copy()
    mat = mat.loc[sig.index.intersection(mat.index)]
    zscore = mat.subtract(mat.mean(axis=1), axis=0).divide(mat.std(axis=1).replace(0, 1), axis=0)

    # Sort samples: Controls first
    conditions = metadata.loc[zscore.columns, "condition"]
    sample_order = (
        conditions[conditions == "Control"].index.tolist() +
        conditions[conditions == "IBD"].index.tolist()
    )
    zscore = zscore[sample_order]

    col_colors = pd.Series(
        [PALETTE.get(c, "gray") for c in conditions[sample_order]],
        index=sample_order, name="Condition",
    )

    g = sns.clustermap(
        zscore,
        cmap="RdBu_r", center=0, vmin=-3, vmax=3,
        col_colors=col_colors,
        col_cluster=False,
        row_cluster=True,
        figsize=(11, 12),
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Z-score"},
        linewidths=0.0,
    )
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=6, rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=8)
    g.figure.suptitle(f"Top {len(zscore)} Differentially Expressed Genes", y=1.01,
                      fontsize=13, fontweight="bold")

    patches = [mpatches.Patch(color=c, label=k) for k, c in PALETTE.items()]
    g.ax_col_dendrogram.legend(handles=patches, loc="center left",
                               bbox_to_anchor=(1.05, 0.5), frameon=False)

    g.figure.savefig(out_dir / "fig6_heatmap_deg.png", dpi=DPI, bbox_inches="tight")
    plt.close(g.figure)
    logger.info("Saved fig6_heatmap_deg.png")


# ── Figure 7: Dispersion ──────────────────────────────────────────────────────

def fig7_dispersion(
    results: pd.DataFrame,
    out_dir: Path,
) -> None:
    df = results.dropna(subset=["baseMean"]).copy()
    df["log2_base"] = np.log2(df["baseMean"] + 1)

    if "dispersions" not in df.columns:
        # Approximate dispersion from LFC SE
        df["dispersions"] = (df.get("lfcSE", pd.Series(np.nan, index=df.index)) ** 2)

    df = df.dropna(subset=["dispersions"])
    if len(df) == 0:
        logger.warning("No dispersion data available; skipping fig7")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    ax.scatter(
        df["log2_base"], df["dispersions"],
        c=df["significant"].map({True: "#E74C3C", False: "#BDC3C7"}),
        s=5, alpha=0.35,
    )

    # Trend line
    valid = df[np.isfinite(df["dispersions"]) & np.isfinite(df["log2_base"])]
    if len(valid) > 20:
        from scipy.stats import binned_statistic
        bstat, bedge, _ = binned_statistic(valid["log2_base"], valid["dispersions"],
                                           statistic="median", bins=20)
        bcentre = (bedge[:-1] + bedge[1:]) / 2
        mask = np.isfinite(bstat)
        ax.plot(bcentre[mask], bstat[mask], color="black", lw=2, label="Median trend")

    ax.set_xlabel("log₂(Base Mean + 1)", fontsize=11)
    ax.set_ylabel("Estimated Dispersion", fontsize=11)
    ax.set_title("Mean-Dispersion Relationship", fontsize=13, fontweight="bold")
    ax.set_ylim(bottom=0)

    patches = [
        mpatches.Patch(color="#E74C3C", label="Significant"),
        mpatches.Patch(color="#BDC3C7", label="Not significant"),
    ]
    ax.legend(handles=patches, fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir, "fig7_dispersion.png")


# ── Figure 8: Pathway enrichment ──────────────────────────────────────────────

def fig8_pathway_enrichment(
    enrich_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    if enrich_df is None or len(enrich_df) == 0:
        logger.warning("No enrichment results; skipping fig8")
        return

    df = enrich_df.copy()
    df["-log10_padj"] = -np.log10(df["padj"].clip(lower=1e-20))
    df = df.sort_values("-log10_padj", ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#E74C3C" if p < 0.05 else "#95A5A6" for p in df["padj"]]
    bars = ax.barh(df["pathway"], df["-log10_padj"], color=colors, edgecolor="white")

    ax.axvline(-np.log10(0.05), color="black", lw=1.0, ls="--", label="padj = 0.05")

    # Overlap count labels
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(
            bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            f"n={int(row['n_overlap'])}", va="center", fontsize=8,
        )

    ax.set_xlabel("-log₁₀(adjusted p-value)", fontsize=11)
    ax.set_title("Pathway Enrichment Analysis (Fisher's Exact Test)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir, "fig8_pathway_enrichment.png")


# ── Figure 9: DE summary ──────────────────────────────────────────────────────

def fig9_de_summary(
    results: pd.DataFrame,
    metadata: pd.DataFrame,
    out_dir: Path,
    padj_thr: float = 0.05,
) -> None:
    n_ctrl = (metadata["condition"] == "Control").sum()
    n_ibd  = (metadata["condition"] == "IBD").sum()

    n_total    = len(results)
    n_tested   = results["padj"].notna().sum()
    sig        = results[results["significant"]]
    n_up       = (sig["direction"] == "Up").sum()
    n_down     = (sig["direction"] == "Down").sum()

    thresholds = [0.05, 0.01, 0.001]
    lfc_bins   = [0, 1, 2, 3]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Differential Expression Summary: IBD vs Control",
                 fontsize=14, fontweight="bold", y=1.02)

    # Panel A: up/down/NS donut
    ax = axes[0]
    n_ns   = n_tested - n_up - n_down
    wedges, texts, autotexts = ax.pie(
        [n_up, n_down, n_ns],
        labels=["Up", "Down", "NS"],
        colors=[PALETTE_DIR["Up"], PALETTE_DIR["Down"], PALETTE_DIR["NS"]],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.5),
    )
    ax.set_title(f"DE Gene Summary\n(n={n_tested:,} tested)", fontsize=11)
    for t in autotexts:
        t.set_fontsize(9)

    # Panel B: padj threshold sensitivity
    ax = axes[1]
    counts_by_thr = [
        (results["padj"] < t).sum()
        for t in thresholds
    ]
    bars = ax.bar(
        [f"padj<{t}" for t in thresholds],
        counts_by_thr,
        color=["#2C3E50", "#2980B9", "#85C1E9"],
        edgecolor="white",
    )
    for b, c in zip(bars, counts_by_thr):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5,
                f"{c:,}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Number of Significant Genes")
    ax.set_title("Sensitivity to padj Threshold", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel C: |LFC| distribution among significant genes
    ax = axes[2]
    lfc_abs = sig["log2FoldChange"].abs()
    n_lfc = [((lfc_abs >= lo) & (lfc_abs < (hi if hi != lfc_bins[-1] else 99))).sum()
             for lo, hi in zip(lfc_bins[:-1], lfc_bins[1:])]
    labels_lfc = [f"|LFC| ∈ [{lo},{hi})" for lo, hi in zip(lfc_bins[:-1], lfc_bins[1:])]
    labels_lfc[-1] = labels_lfc[-1].replace("3)", "∞)")
    bars = ax.bar(labels_lfc, n_lfc, color=["#E8DAEF", "#BB8FCE", "#7D3C98"],
                  edgecolor="white")
    for b, c in zip(bars, n_lfc):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                str(c), ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Number of Significant Genes")
    ax.set_title("|log₂FC| Distribution\nAmong Significant Genes", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, out_dir, "fig9_de_summary.png")


# ── Driver: generate all figures ──────────────────────────────────────────────

def generate_all(
    counts: pd.DataFrame,
    vst_counts: pd.DataFrame,
    results: pd.DataFrame,
    metadata: pd.DataFrame,
    enrich_df: pd.DataFrame,
    out_dir: str | Path,
    cfg: Optional[dict] = None,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = cfg or {}

    padj_thr = cfg.get("padj_threshold", 0.05)
    lfc_thr  = cfg.get("lfc_threshold",  1.0)

    logger.info("Generating figures → %s", out)
    fig1_sample_qc(counts, metadata, out)
    fig2_pca(vst_counts, metadata, out)
    fig3_sample_distance(vst_counts, metadata, out)
    fig4_ma_plot(results, out, padj_thr, lfc_thr)
    fig5_volcano(results, out, padj_thr, lfc_thr)
    fig6_heatmap(vst_counts, results, metadata, out)
    fig7_dispersion(results, out)
    fig8_pathway_enrichment(enrich_df, out)
    fig9_de_summary(results, metadata, out, padj_thr)
    logger.info("All 9 figures saved.")
