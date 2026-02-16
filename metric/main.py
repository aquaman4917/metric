"""
Brain Control Architecture Pipeline — main entry point.

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

import os
import sys
import time
import argparse
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from modules.utils import load_config, override_config, param_string, ensure_dir, setup_logging
from modules.loader import load_connectivity, filter_subregion
from modules.metrics import compute_all_metrics, METRIC_REGISTRY
from modules import analysis
from modules import plotting

logger = logging.getLogger("metric.pipeline")


# ================================================================
# Single pipeline run
# ================================================================

def run_pipeline(cfg: dict) -> dict:
    """
    Execute full pipeline for one configuration.

    Steps:
        1. Load data + optional subregion filter
        2. Compute metrics per subject (parallel)
        3. Build results DataFrame + save CSV
        4. Lifespan trend analysis
        5. Age bin analysis + stat tests
        6. Generate all figures

    Returns:
        dict with keys: df, trend, bin_stats, stat_results, cfg
    """
    pstr = param_string(cfg)
    out_dir = ensure_dir(os.path.join(cfg['paths']['output_dir'], pstr))
    fig_dir = ensure_dir(os.path.join(out_dir, 'figures'))

    logger.info("=" * 60)
    logger.info("METRIC PIPELINE")
    logger.info("  Run:    %s", pstr)
    logger.info("  Input:  %s", cfg['paths']['input_mat'])
    logger.info("  Output: %s", out_dir)
    logger.info("=" * 60)

    # --- 1. Load ---
    logger.info("=" * 60)
    logger.info("DATA LOADING")
    logger.info("=" * 60)
    age, conn_list, n_subj = load_connectivity(cfg)

    # Subregion
    sub_nodes = cfg['subregion'].get('nodes')
    if sub_nodes is not None:
        conn_list = filter_subregion(conn_list, sub_nodes)
        cfg['network']['net_size'] = len(sub_nodes)

    # --- 2. Compute metrics ---
    logger.info("=" * 60)
    logger.info("METRIC COMPUTATION")
    logger.info("=" * 60)
    logger.info(f"[compute] {n_subj} subjects, parallel={cfg['compute']['parallel']}")
    t0 = time.time()

    if cfg['compute']['parallel']:
        results_list = Parallel(n_jobs=cfg['compute']['n_jobs'], verbose=0)(
            delayed(_compute_one)(conn_list[i], cfg, i)
            for i in tqdm(range(n_subj), desc="Computing metrics")
        )
    else:
        results_list = []
        for i in tqdm(range(n_subj), desc="Computing metrics"):
            results_list.append(_compute_one(conn_list[i], cfg, i))

    elapsed = time.time() - t0
    logger.info(f"[compute] Done: {elapsed:.1f}s ({elapsed/n_subj:.3f}s/subj)")

    # --- 3. Build DataFrame ---
    df = pd.DataFrame(results_list)

    # Age columns
    if cfg['age']['unit'] == 'month':
        df.insert(0, 'age_months', age)
        df.insert(1, 'age_years', age / 12.0)
    else:
        df.insert(0, 'age_years', age)
        df.insert(1, 'age_months', age * 12.0)

    # Save CSV
    csv_path = os.path.join(out_dir, f"results_{pstr}.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"[output] CSV: {csv_path}")

    # --- Summary ---
    active_metrics = _active_metrics(cfg)
    _print_summary(df, active_metrics, n_subj)

    # --- 4. Lifespan trend ---
    logger.info("--- Lifespan Trend ---")
    trend_df = analysis.lifespan_trend(df, active_metrics, cfg)

    trend_csv = os.path.join(out_dir, f"trend_{pstr}.csv")
    trend_df.to_csv(trend_csv, index=False)

    for _, row in trend_df.iterrows():
        plotting.plot_scatter_trend(df, row['metric'], row.to_dict(), cfg, fig_dir, pstr)

    # --- 5. Age bin ---
    logger.info("--- Age Bin Analysis ---")
    df = analysis.assign_age_bins(df, cfg)

    bin_stats_df = analysis.bin_summary_stats(df, active_metrics)
    bin_stats_csv = os.path.join(out_dir, f"bin_stats_{pstr}.csv")
    bin_stats_df.to_csv(bin_stats_csv, index=False)

    # Stat tests
    stat_results = analysis.stat_tests(df, active_metrics, cfg)

    stat_rows = []
    for mname, res in stat_results.items():
        for pw in res['pairwise']:
            stat_rows.append({'metric': mname, **pw})
    if stat_rows:
        stat_csv = os.path.join(out_dir, f"pairwise_stats_{pstr}.csv")
        pd.DataFrame(stat_rows).to_csv(stat_csv, index=False)

    # Bin figures
    for mname in active_metrics:
        sr = stat_results.get(mname)
        plotting.plot_boxplot_bins(df, mname, sr, cfg, fig_dir, pstr)

    # Heatmap
    plotting.plot_heatmap_summary(bin_stats_df, cfg, fig_dir, pstr)

    # --- 6. Metric comparison plots ---
    comparison_pairs = [
        ('DC', 'OCA'), ('DC', 'DC_PC'),
        ('OCA_P', 'OCA_C'), ('DC_PC', 'Prov_ratio'),
    ]
    for m1, m2 in comparison_pairs:
        if m1 in active_metrics and m2 in active_metrics:
            plotting.plot_metric_comparison(df, m1, m2, cfg, fig_dir, pstr)

    # --- Done ---
    logger.info(f"[output] Results: {out_dir}")
    logger.info(f"[output] Figures: {fig_dir}")

    return {
        'df': df,
        'trend': trend_df,
        'bin_stats': bin_stats_df,
        'stat_results': stat_results,
        'cfg': cfg,
    }


def _compute_one(conn: np.ndarray, cfg: dict, idx: int) -> dict:
    """Compute all metrics for a single subject. (Worker function for joblib)"""
    try:
        return compute_all_metrics(conn, cfg)
    except Exception as e:
        logger.error(f"Subject {idx} failed: {e}")
        return {m: np.nan for m in _active_metrics(cfg)}


def _active_metrics(cfg: dict) -> list:
    """Return list of enabled metric names."""
    return [m for m, enabled in cfg['metrics'].items() if enabled]


def _print_summary(df: pd.DataFrame, metric_names: list, n_subj: int):
    """Print summary statistics."""
    logger.info(f"\n{'='*50}")
    logger.info(f"  Summary: {n_subj} subjects, "
                f"age {df['age_years'].min():.1f}~{df['age_years'].max():.1f} years")
    logger.info(f"{'='*50}")
    for mname in metric_names:
        if mname not in df.columns:
            continue
        v = df[mname].replace([np.inf, -np.inf], np.nan).dropna()
        logger.info(f"  {mname:12s}: mean={v.mean():.4f}  std={v.std():.4f}  "
                     f"median={v.median():.4f}  (n={len(v)})")


# ================================================================
# Batch mode
# ================================================================

def run_batch_sweep(base_cfg: dict):
    """Sweep over density × frac combinations."""
    densities = [0.10, 0.15, 0.20, 0.25, 0.30]
    fracs = [0.03, 0.05, 0.10, 0.15, 0.20]

    logger.info(f"\n{'#'*60}")
    logger.info(f"  BATCH SWEEP: {len(densities)} densities × {len(fracs)} fracs")
    logger.info(f"{'#'*60}")

    all_results = []
    for d in densities:
        for f in fracs:
            cfg = override_config(base_cfg, {
                'network': {'density': d, 'frac': f}
            })
            try:
                res = run_pipeline(cfg)
                all_results.append(res)
            except Exception as e:
                logger.error(f"[BATCH] density={d}, frac={f} FAILED: {e}")

    # Summary table
    if all_results:
        summary = _batch_summary(all_results)
        summary_path = os.path.join(base_cfg['paths']['output_dir'], 'batch_summary.csv')
        summary.to_csv(summary_path, index=False)
        logger.info(f"\nBatch summary → {summary_path}")
        print(summary.to_string(index=False))


def run_batch_subregions(base_cfg: dict, subregion_map: dict):
    """
    Run pipeline for each subregion.

    subregion_map: {'Frontal': [0,1,...,51], 'Temporal': [52,...], ...}
    """
    logger.info(f"\n{'#'*60}")
    logger.info(f"  SUBREGION ANALYSIS: {len(subregion_map)} regions")
    logger.info(f"{'#'*60}")

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
            logger.error(f"[SUBREGION] {label} FAILED: {e}")

    if all_results:
        summary = _batch_summary(all_results)
        summary_path = os.path.join(base_cfg['paths']['output_dir'], 'subregion_summary.csv')
        summary.to_csv(summary_path, index=False)
        logger.info(f"\nSubregion summary → {summary_path}")


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
        for mname in _active_metrics(c):
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
