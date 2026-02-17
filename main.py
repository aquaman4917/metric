"""
Brain Control Architecture Pipeline — main entry point.

The pipeline is a strict 4-stage chain:

    Stage 1  Preprocessing   →  PreprocessedData
    Stage 2  Metrics          →  pd.DataFrame
    Stage 3  Analysis         →  AnalysisResults
    Stage 4  Visualization    →  figures + CSVs

Each stage is self-contained in ``modules/stage{N}_*.py`` and can be
developed, tested, or replaced independently.

Usage:
    # Default config
    python main.py

    # Custom config
    python main.py --config configs/my_config.yaml

    # Override parameters
    python main.py --density 0.30 --frac 0.10

    # Batch: sweep density × frac
    python main.py --batch-sweep

    # Subregion analysis
    python main.py --subregion Frontal
"""

import argparse
import logging

import numpy as np
import pandas as pd

from modules.utils import (
    load_config,
    override_config,
    active_metrics,
    setup_run_dirs,
    print_metric_summary,
    setup_logging,
    param_string,
)
from modules.stage1_preprocessing import Stage1Preprocessing
from modules.stage2_metrics import Stage2Metrics
from modules.stage3_analysis import Stage3Analysis
from modules.stage4_visualization import Stage4Visualization

logger = logging.getLogger("metric.pipeline")


# ================================================================
# Single pipeline run
# ================================================================

def run_pipeline(cfg: dict) -> dict:
    """
    Execute the full 4-stage pipeline for one configuration.

    Returns:
        dict with keys: df, trend, bin_stats, stat_results, qc_meta, cfg
    """
    dirs = setup_run_dirs(cfg)
    pstr = dirs['param_str']
    metrics_list = active_metrics(cfg)

    logger.info("=" * 60)
    logger.info("METRIC PIPELINE")
    logger.info("  Run dir : %s", dirs['run_dir'])
    logger.info("  Input   : %s", cfg['paths']['input_mat'])
    logger.info("  Metrics : %s", ", ".join(metrics_list))
    logger.info("=" * 60)

    # ---- Stage 1: Preprocessing ----
    preprocessed = Stage1Preprocessing.run(cfg)

    # ---- Stage 2: Metric computation ----
    df = Stage2Metrics.run(cfg, preprocessed)

    # Save raw CSV (before QC)
    raw_csv = dirs['run_dir'] + f"/results_raw_{pstr}.csv"
    df.to_csv(raw_csv, index=False)
    logger.info("[output] Raw CSV: %s", raw_csv)

    # ---- Stage 3: Analysis (QC + trends + bins + stats) ----
    analysis_results = Stage3Analysis.run(cfg, df, metrics_list)

    # Persist analysis outputs
    Stage3Analysis(cfg, df, metrics_list).save_results(
        analysis_results, dirs['run_dir'], pstr,
    )

    # Also save the post-QC DataFrame
    clean_csv = dirs['run_dir'] + f"/results_{pstr}.csv"
    analysis_results.df.to_csv(clean_csv, index=False)
    logger.info("[output] Clean CSV: %s", clean_csv)

    print_metric_summary(analysis_results.df, metrics_list)

    # ---- Stage 4: Visualization ----
    Stage4Visualization.run(cfg, analysis_results, dirs['fig_dir'], pstr)

    # ---- Done ----
    logger.info("[output] Results  : %s", dirs['run_dir'])
    logger.info("[output] Figures  : %s", dirs['fig_dir'])

    return {
        'df': analysis_results.df,
        'trend': analysis_results.trend_df,
        'bin_stats': analysis_results.bin_stats_df,
        'stat_results': analysis_results.stat_results,
        'qc_meta': analysis_results.qc_meta,
        'cfg': cfg,
    }


# ================================================================
# Batch modes
# ================================================================

def run_batch_sweep(base_cfg: dict):
    """Sweep over density × frac combinations."""
    densities = [0.10, 0.15, 0.20, 0.25, 0.30]
    fracs = [0.03, 0.05, 0.10, 0.15, 0.20]

    logger.info("#" * 60)
    logger.info("  BATCH SWEEP: %d densities × %d fracs", len(densities), len(fracs))
    logger.info("#" * 60)

    all_results = []
    for d in densities:
        for f in fracs:
            cfg = override_config(base_cfg, {'network': {'density': d, 'frac': f}})
            try:
                res = run_pipeline(cfg)
                all_results.append(res)
            except Exception as e:
                logger.error("[BATCH] density=%s, frac=%s FAILED: %s", d, f, e)

    if all_results:
        summary = _batch_summary(all_results)
        import os
        summary_path = os.path.join(base_cfg['paths']['output_dir'], 'batch_summary.csv')
        summary.to_csv(summary_path, index=False)
        logger.info("Batch summary → %s", summary_path)
        print(summary.to_string(index=False))


def run_batch_subregions(base_cfg: dict, subregion_map: dict):
    """Run pipeline for each subregion."""
    logger.info("#" * 60)
    logger.info("  SUBREGION ANALYSIS: %d regions", len(subregion_map))
    logger.info("#" * 60)

    all_results = []
    for label, nodes in subregion_map.items():
        cfg = override_config(base_cfg, {
            'subregion': {'nodes': nodes, 'label': label},
            'network': {'net_size': len(nodes)},
        })
        try:
            res = run_pipeline(cfg)
            all_results.append(res)
        except Exception as e:
            logger.error("[SUBREGION] %s FAILED: %s", label, e)

    if all_results:
        summary = _batch_summary(all_results)
        import os
        summary_path = os.path.join(base_cfg['paths']['output_dir'], 'subregion_summary.csv')
        summary.to_csv(summary_path, index=False)
        logger.info("Subregion summary → %s", summary_path)


def _batch_summary(results_list: list) -> pd.DataFrame:
    """Create summary DataFrame from multiple pipeline runs."""
    rows = []
    for res in results_list:
        c = res['cfg']
        d = res['df']
        row = {
            'label': param_string(c),
            'density': c['network']['density'],
            'frac': c['network']['frac'],
            'subregion': c['subregion']['label'],
            'n_subj': len(d),
        }
        for mname in active_metrics(c):
            if mname in d.columns:
                v = d[mname].replace([np.inf, -np.inf], np.nan).dropna()
                row[f'{mname}_mean'] = v.mean()
                row[f'{mname}_std'] = v.std()
        rows.append(row)
    return pd.DataFrame(rows)


# ================================================================
# CLI
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Brain Control Architecture Pipeline")
    parser.add_argument('--config', default='configs/default.yaml',
                        help='Path to YAML config')
    parser.add_argument('--density', type=float, default=None)
    parser.add_argument('--frac', type=float, default=None)
    parser.add_argument('--input', type=str, default=None,
                        help='Override input MAT (path, filename, or dataset name)')
    parser.add_argument('--subregion', type=str, default=None,
                        help='Subregion label (requires BNA_subregions.xlsx)')
    parser.add_argument('--batch-sweep', action='store_true',
                        help='Run density×frac sweep')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel processing')
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    # Load base config
    cfg = load_config(args.config)

    # CLI overrides
    overrides = {}
    if args.density is not None:
        overrides.setdefault('network', {})['density'] = args.density
    if args.frac is not None:
        overrides.setdefault('network', {})['frac'] = args.frac
    if args.input is not None:
        overrides.setdefault('paths', {})['input_mat'] = args.input
    if args.no_parallel:
        overrides.setdefault('compute', {})['parallel'] = False
    if overrides:
        cfg = override_config(cfg, overrides)

    # Batch or single
    if args.batch_sweep:
        run_batch_sweep(cfg)
    else:
        run_pipeline(cfg)


if __name__ == '__main__':
    main()
