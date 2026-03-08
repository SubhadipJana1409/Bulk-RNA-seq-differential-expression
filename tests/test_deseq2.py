"""Tests for src/models/deseq2.py"""
import pytest
from src.data.simulator import simulate_counts
from src.models.deseq2 import DEAnalysis


@pytest.fixture(scope="module")
def small_de():
    """Run a small DE analysis once for all tests in this module."""
    counts, meta = simulate_counts(n_samples=12, n_genes=300, seed=42)
    de = DEAnalysis(counts, meta, cfg={"min_count": 5, "min_samples": 2})
    de.run()
    return de


class TestDEAnalysis:
    def test_results_not_none(self, small_de):
        assert small_de.results is not None

    def test_results_columns(self, small_de):
        expected = {"baseMean", "log2FoldChange", "pvalue", "padj", "significant", "direction"}
        assert expected.issubset(set(small_de.results.columns))

    def test_padj_between_0_and_1(self, small_de):
        valid = small_de.results["padj"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_direction_values(self, small_de):
        valid_dirs = {"Up", "Down", "NS"}
        assert set(small_de.results["direction"].unique()).issubset(valid_dirs)

    def test_significant_is_boolean(self, small_de):
        assert small_de.results["significant"].dtype == bool

    def test_normalized_counts_shape(self, small_de):
        assert small_de.normalized_counts is not None
        nc = small_de.normalized_counts
        assert nc.shape[1] == 12  # samples

    def test_normalized_counts_positive(self, small_de):
        assert (small_de.normalized_counts.values >= 0).all()

    def test_vst_counts_log_scale(self, small_de):
        # VST = log2(norm+1), so all values ≥ 0
        assert (small_de.vst_counts.values >= 0).all()

    def test_get_sig_genes(self, small_de):
        sig = small_de.get_sig_genes()
        assert sig["significant"].all()

    def test_get_sig_genes_direction_filter(self, small_de):
        up = small_de.get_sig_genes(direction="Up")
        assert (up["direction"] == "Up").all()

    def test_get_top_genes(self, small_de):
        top = small_de.get_top_genes(n=10)
        assert len(top) <= 10

    def test_save_results(self, small_de, tmp_path):
        small_de.save_results(tmp_path)
        assert (tmp_path / "de_results_all.csv").exists()
        assert (tmp_path / "de_results_significant.csv").exists()
        assert (tmp_path / "normalized_counts.csv").exists()

    def test_filter_low_counts(self):
        counts, meta = simulate_counts(n_samples=8, n_genes=200, seed=0)
        # Add some all-zero rows
        counts.iloc[:10] = 0
        de = DEAnalysis(counts, meta)
        filtered = de.filter_low_counts(min_count=1, min_samples=4)
        assert len(filtered) < len(counts)

    def test_run_raises_before_run(self):
        counts, meta = simulate_counts(n_samples=8, n_genes=100, seed=0)
        de = DEAnalysis(counts, meta)
        with pytest.raises(RuntimeError):
            de.get_sig_genes()
