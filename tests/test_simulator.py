"""Tests for src/data/simulator.py"""
import numpy as np
import pandas as pd
import pytest
from src.data.simulator import simulate_counts, load_or_simulate, IBD_UP_GENES, IBD_DOWN_GENES


class TestSimulateCounts:
    def test_output_shapes(self):
        counts, meta = simulate_counts(n_samples=10, n_genes=200, seed=0)
        assert counts.shape[1] == 10
        assert len(meta) == 10
        assert counts.shape[0] >= 200  # may include named genes

    def test_condition_split(self):
        counts, meta = simulate_counts(n_samples=10, n_ibd=4, n_genes=100, seed=0)
        assert (meta["condition"] == "IBD").sum() == 4
        assert (meta["condition"] == "Control").sum() == 6

    def test_no_negative_counts(self):
        counts, _ = simulate_counts(n_samples=8, n_genes=100, seed=1)
        assert (counts.values >= 0).all()

    def test_integer_counts(self):
        counts, _ = simulate_counts(n_samples=8, n_genes=100, seed=2)
        assert counts.dtypes.apply(lambda d: np.issubdtype(d, np.integer)).all()

    def test_ibd_up_genes_higher(self):
        """IBD_UP genes should on average be higher in IBD than Control."""
        counts, meta = simulate_counts(n_samples=20, n_genes=200, seed=42)
        ibd_cols  = meta[meta["condition"] == "IBD"].index
        ctrl_cols = meta[meta["condition"] == "Control"].index
        up_genes  = [g for g in IBD_UP_GENES if g in counts.index]
        assert len(up_genes) > 0
        ibd_mean  = counts.loc[up_genes, ibd_cols].mean().mean()
        ctrl_mean = counts.loc[up_genes, ctrl_cols].mean().mean()
        assert ibd_mean > ctrl_mean

    def test_ibd_down_genes_lower(self):
        """IBD_DOWN genes should on average be lower in IBD than Control."""
        counts, meta = simulate_counts(n_samples=20, n_genes=200, seed=42)
        ibd_cols  = meta[meta["condition"] == "IBD"].index
        ctrl_cols = meta[meta["condition"] == "Control"].index
        dn_genes  = [g for g in IBD_DOWN_GENES if g in counts.index]
        assert len(dn_genes) > 0
        ibd_mean  = counts.loc[dn_genes, ibd_cols].mean().mean()
        ctrl_mean = counts.loc[dn_genes, ctrl_cols].mean().mean()
        assert ibd_mean < ctrl_mean

    def test_metadata_columns(self):
        _, meta = simulate_counts(n_samples=6, n_genes=50, seed=0)
        for col in ["condition", "batch", "age", "sex"]:
            assert col in meta.columns

    def test_load_or_simulate_no_paths(self):
        counts, meta = load_or_simulate({"n_samples": 8, "n_genes": 100, "seed": 5})
        assert counts.shape[1] == 8

    def test_reproducibility(self):
        c1, _ = simulate_counts(n_samples=8, n_genes=100, seed=7)
        c2, _ = simulate_counts(n_samples=8, n_genes=100, seed=7)
        pd.testing.assert_frame_equal(c1, c2)
