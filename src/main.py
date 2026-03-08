"""
Day 21 · Bulk RNA-seq Differential Expression Pipeline
=======================================================
IBD vs Control · PyDESeq2 · 9 publication figures

Usage
-----
    python -m src.main                        # simulate data, run full pipeline
    python -m src.main --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


# ── Bootstrap path ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.simulator    import load_or_simulate  # noqa: E402
from src.models.deseq2     import DEAnalysis  # noqa: E402
from src.models.enrichment import run_enrichment  # noqa: E402
from src.visualization.plots import generate_all  # noqa: E402
from src.utils.logger      import setup_logging  # noqa: E402
from src.utils.config      import load_config  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bulk RNA-seq DE pipeline")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--quiet",  action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)
    setup_logging(level=logging.WARNING if args.quiet else logging.INFO)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Day 21 · Bulk RNA-seq DE Pipeline")
    logger.info("=" * 60)

    # ── Step 1: Load / simulate data ─────────────────────────────────────────
    logger.info("[1/6] Loading count data …")
    counts, metadata = load_or_simulate(cfg.get("data", {}))
    logger.info(
        "Count matrix: %d genes × %d samples  (%d IBD, %d Control)",
        len(counts), len(counts.columns),
        (metadata["condition"] == "IBD").sum(),
        (metadata["condition"] == "Control").sum(),
    )

    # ── Step 2: Differential expression ──────────────────────────────────────
    logger.info("[2/6] Running DESeq2 …")
    de = DEAnalysis(counts, metadata, cfg=cfg.get("de", {}))
    results = de.run(
        reference_level=cfg.get("de", {}).get("reference_level", "Control"),
        n_cpus=cfg.get("de", {}).get("n_cpus", 1),
    )

    vst_counts = de.vst_counts

    # ── Step 3: Pathway enrichment ────────────────────────────────────────────
    logger.info("[3/6] Pathway enrichment …")
    enrich_df = run_enrichment(
        results,
        gene_universe=results.index.tolist(),
        padj_threshold=cfg.get("de", {}).get("padj_threshold", 0.05),
        lfc_threshold=cfg.get("de", {}).get("lfc_threshold", 1.0),
    )

    # ── Step 4: Save result tables ────────────────────────────────────────────
    logger.info("[4/6] Saving result tables …")
    de.save_results(out)
    metadata.to_csv(out / "sample_metadata.csv")
    if len(enrich_df) > 0:
        enrich_df.to_csv(out / "pathway_enrichment.csv", index=False)

    # ── Step 5: Generate figures ──────────────────────────────────────────────
    logger.info("[5/6] Generating figures …")
    generate_all(
        counts=counts,
        vst_counts=vst_counts,
        results=results,
        metadata=metadata,
        enrich_df=enrich_df,
        out_dir=out,
        cfg=cfg.get("de", {}),
    )

    # ── Step 6: Print summary ─────────────────────────────────────────────────
    sig = results[results["significant"]]
    n_up = (sig["direction"] == "Up").sum()
    n_dn = (sig["direction"] == "Down").sum()
    elapsed = time.time() - t0

    logger.info("[6/6] Done.")
    print("\n" + "=" * 50)
    print("  Day 21 · Bulk RNA-seq DE Summary")
    print("=" * 50)
    print(f"  Samples   : {len(metadata)} ({(metadata['condition']=='IBD').sum()} IBD, "
          f"{(metadata['condition']=='Control').sum()} Control)")
    print(f"  Genes     : {len(results):,} tested after filtering")
    print(f"  Sig genes : {len(sig):,} (padj<0.05, |lfc|≥1)")
    print(f"  Up        : {n_up:,} (higher in IBD)")
    print(f"  Down      : {n_dn:,} (lower in IBD)")
    if len(enrich_df) > 0:
        enriched = enrich_df[enrich_df["padj"] < 0.05]
        print(f"  Pathways  : {len(enriched)}/{len(enrich_df)} enriched (padj<0.05)")
    print(f"  Figures   : 9 saved to {out}/")
    print(f"  Elapsed   : {elapsed:.1f}s")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
