"""Tests for enrichment and visualization."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from src.data.simulator import simulate_counts
from src.models.deseq2 import DEAnalysis
from src.models.enrichment import run_enrichment, GENE_SETS
from src.visualization.plots import (
    fig1_sample_qc, fig2_pca, fig4_ma_plot, fig5_volcano,
    fig7_dispersion, fig8_pathway_enrichment, fig9_de_summary,
)


@pytest.fixture(scope="module")
def de_results():
    counts, meta = simulate_counts(n_samples=12, n_genes=300, seed=99)
    de = DEAnalysis(counts, meta, cfg={"min_count": 5, "min_samples": 2})
    de.run()
    return de


class TestEnrichment:
    def test_returns_dataframe(self, de_results):
        enrich = run_enrichment(de_results.results)
        assert isinstance(enrich, pd.DataFrame)

    def test_columns_present(self, de_results):
        enrich = run_enrichment(de_results.results)
        for col in ["pathway", "n_overlap", "odds_ratio", "pvalue", "padj"]:
            assert col in enrich.columns

    def test_pvalue_range(self, de_results):
        enrich = run_enrichment(de_results.results)
        assert (enrich["pvalue"] >= 0).all() and (enrich["pvalue"] <= 1).all()

    def test_padj_range(self, de_results):
        enrich = run_enrichment(de_results.results)
        assert (enrich["padj"] >= 0).all() and (enrich["padj"] <= 1).all()

    def test_gene_sets_non_empty(self):
        assert len(GENE_SETS) >= 8
        for name, genes in GENE_SETS.items():
            assert len(genes) >= 5, f"Gene set '{name}' too small"

    def test_empty_de_results(self):
        """Should handle zero significant genes gracefully."""
        counts, meta = simulate_counts(n_samples=6, n_genes=50, seed=0)
        results = pd.DataFrame({
            "baseMean": [10] * 50,
            "log2FoldChange": [0.1] * 50,
            "pvalue": [0.9] * 50,
            "padj": [0.99] * 50,
            "significant": [False] * 50,
            "direction": ["NS"] * 50,
        }, index=[f"GENE_{i}" for i in range(50)])
        enrich = run_enrichment(results)
        assert isinstance(enrich, pd.DataFrame)


class TestPlots:
    def test_fig1_creates_file(self, de_results, tmp_path):
        counts, meta = simulate_counts(n_samples=12, n_genes=300, seed=99)
        fig1_sample_qc(counts, meta, tmp_path)
        assert (tmp_path / "fig1_sample_qc.png").exists()

    def test_fig2_creates_file(self, de_results, tmp_path):
        counts, meta = simulate_counts(n_samples=12, n_genes=300, seed=99)
        de = DEAnalysis(counts, meta, cfg={"min_count": 5, "min_samples": 2})
        de.run()
        fig2_pca(de.vst_counts, meta, tmp_path)
        assert (tmp_path / "fig2_pca.png").exists()

    def test_fig4_creates_file(self, de_results, tmp_path):
        fig4_ma_plot(de_results.results, tmp_path)
        assert (tmp_path / "fig4_ma_plot.png").exists()

    def test_fig5_creates_file(self, de_results, tmp_path):
        fig5_volcano(de_results.results, tmp_path)
        assert (tmp_path / "fig5_volcano.png").exists()

    def test_fig7_creates_file(self, de_results, tmp_path):
        fig7_dispersion(de_results.results, tmp_path)
        assert (tmp_path / "fig7_dispersion.png").exists()

    def test_fig8_creates_file(self, de_results, tmp_path):
        enrich = run_enrichment(de_results.results)
        fig8_pathway_enrichment(enrich, tmp_path)
        assert (tmp_path / "fig8_pathway_enrichment.png").exists()

    def test_fig9_creates_file(self, de_results, tmp_path):
        counts, meta = simulate_counts(n_samples=12, n_genes=300, seed=99)
        fig9_de_summary(de_results.results, meta, tmp_path)
        assert (tmp_path / "fig9_de_summary.png").exists()

    def test_fig8_handles_empty(self, tmp_path):
        """Empty enrichment should not crash."""
        fig8_pathway_enrichment(pd.DataFrame(), tmp_path)
        assert not (tmp_path / "fig8_pathway_enrichment.png").exists()
