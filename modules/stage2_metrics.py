"""
Stage 2: Metric Computation
============================
Pipeline:
  1. Load preprocessed network features
  2. Compute all enabled metrics per subject
  3. Build results DataFrame
  4. Save CSV results
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List
from tqdm import tqdm
from joblib import Parallel, delayed

from .stage1_preprocessing import PreprocessedData
from .metrics import MetricComputer, MetricRegistry, get_metric_registry
from .mds_conditional_zscore import MDSConditionalZScore
from .utils import active_metrics as _active_metrics, build_age_columns, print_metric_summary

logger = logging.getLogger(__name__)


# ================================================================
# Module-level worker (picklable — no bound methods)
# ================================================================

def _worker(adj: np.ndarray, mdset: np.ndarray,
            top_nodes: np.ndarray, cfg: dict) -> Dict[str, float]:
    """
    Compute all enabled metrics for one subject.

    This is a module-level function so joblib can pickle it without
    serialising the entire Stage2Metrics / MetricRegistry instance.
    """
    try:
        computer = MetricComputer()
        return computer.compute_all_from_preprocessed(adj, mdset, top_nodes, cfg)
    except Exception as e:
        logger.error(f"Worker failed: {e}")
        return {m: np.nan for m, on in cfg['metrics'].items() if on and not m.endswith('_cz')}


# ================================================================
# Stage 2 Class
# ================================================================

class Stage2Metrics:
    """
    Stage 2: Metric computation.

    Computes all enabled metrics for each subject using preprocessed data.
    """

    def __init__(self, cfg: dict, preprocessed: PreprocessedData):
        self.cfg = cfg
        self.preprocessed = preprocessed

    @classmethod
    def run(cls, cfg: dict, preprocessed: PreprocessedData) -> pd.DataFrame:
        stage = cls(cfg, preprocessed)
        return stage.execute()

    @classmethod
    def list_available_metrics(cls) -> List[str]:
        base_metrics = list(get_metric_registry().get_all().keys())
        derived_cz = [f"{m}_cz" for m in base_metrics if not m.endswith('_cz')]
        return base_metrics + derived_cz

    @classmethod
    def get_enabled_metrics(cls, cfg: dict) -> List[str]:
        return _active_metrics(cfg)

    def execute(self) -> pd.DataFrame:
        logger.info("=" * 60)
        logger.info("  Stage 2: Metric Computation")
        logger.info("=" * 60)

        enabled = self.get_enabled_metrics(self.cfg)
        logger.info(f"[Stage2] Enabled metrics ({len(enabled)}): {', '.join(enabled)}")

        results_list = self._compute_all_subjects()
        df = self._build_dataframe(results_list)
        df = self._apply_mds_conditional_zscore(df)
        self._print_summary(df, enabled)

        logger.info(f"[Stage2] Complete: {len(df)} subjects, {len(enabled)} metrics")
        return df

    def _compute_all_subjects(self) -> List[Dict[str, float]]:
        n_subj = self.preprocessed.n_subjects
        use_parallel = self.cfg['compute']['parallel']

        logger.info(f"[Stage2] Computing metrics for {n_subj} subjects "
                    f"(parallel={use_parallel})...")

        pre = self.preprocessed
        cfg = self.cfg

        if use_parallel:
            n_jobs = cfg['compute']['n_jobs']
            results_list = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(_worker)(
                    pre.conn_binary[i],
                    pre.mdsets[i],
                    pre.top_nodes[i],
                    cfg,
                )
                for i in tqdm(range(n_subj), desc="Computing metrics")
            )
        else:
            results_list = [
                _worker(pre.conn_binary[i], pre.mdsets[i], pre.top_nodes[i], cfg)
                for i in tqdm(range(n_subj), desc="Computing metrics")
            ]

        return results_list

    def _build_dataframe(self, results_list: List[Dict[str, float]]) -> pd.DataFrame:
        logger.info("[Stage2] Building DataFrame...")
        df = pd.DataFrame(results_list)
        return build_age_columns(df, self.preprocessed.age, self.cfg)

    def _apply_mds_conditional_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute cohort-level MDS-conditional z-score metrics (*_cz) when enabled.
        """
        enabled_cz = [
            mname for mname, on in self.cfg.get('metrics', {}).items()
            if on and mname.endswith('_cz')
        ]
        if not enabled_cz:
            return df

        base_metrics = []
        missing_base = []
        for cz_name in enabled_cz:
            base_name = cz_name[:-3]
            if base_name in df.columns:
                base_metrics.append(base_name)
            else:
                missing_base.append((cz_name, base_name))

        if missing_base:
            pairs = ", ".join([f"{cz}->{base}" for cz, base in missing_base])
            logger.warning("[Stage2] Conditional z-score skipped for missing base metrics: %s", pairs)

        if not base_metrics:
            logger.warning("[Stage2] No valid base metrics for conditional z-score; skipping")
            return df

        base_metrics = list(dict.fromkeys(base_metrics))

        cz_cfg = self.cfg.get('stats', {}).get('mds_conditional_zscore', {})
        n_bins = int(cz_cfg.get('n_bins', 10))
        min_bin_n = int(cz_cfg.get('min_bin_n', 20))
        mds_col = str(cz_cfg.get('mds_col', 'MDS_size'))

        if mds_col not in df.columns:
            fallback_col = 'MDS_size_fraction' if mds_col == 'MDS_size' else 'MDS_size'
            if fallback_col in df.columns:
                logger.warning(
                    "[Stage2] mds_col '%s' not found, using '%s' for conditional z-score",
                    mds_col, fallback_col,
                )
                mds_col = fallback_col
            else:
                logger.warning(
                    "[Stage2] No MDS column found for conditional z-score (checked '%s', '%s')",
                    mds_col, fallback_col,
                )
                return df

        logger.info(
            "[Stage2] Applying MDS-conditional z-score: metrics=%s, n_bins=%d, min_bin_n=%d, mds_col=%s",
            base_metrics, n_bins, min_bin_n, mds_col,
        )

        norm = MDSConditionalZScore(
            df,
            metrics=base_metrics,
            n_bins=n_bins,
            min_bin_n=min_bin_n,
            mds_col=mds_col,
        )
        df_out = norm.fit_transform()

        produced = [f"{m}_cz" for m in base_metrics if f"{m}_cz" in df_out.columns]
        logger.info("[Stage2] Added conditional z-score columns: %s", produced)
        return df_out

    def _print_summary(self, df: pd.DataFrame, metric_names: List[str]):
        print_metric_summary(df, metric_names)


# ================================================================
# Standalone Function (Backward Compatibility)
# ================================================================

def run_metrics(cfg: dict, preprocessed: PreprocessedData) -> pd.DataFrame:
    return Stage2Metrics.run(cfg, preprocessed)


def list_available_metrics() -> List[str]:
    return Stage2Metrics.list_available_metrics()
