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

logger = logging.getLogger(__name__)


# ================================================================
# Stage 2 Class
# ================================================================

class Stage2Metrics:
    """
    Stage 2: Metric computation.

    Computes all enabled metrics for each subject using preprocessed data.
    """

    def __init__(self, cfg: dict, preprocessed: PreprocessedData):
        """
        Initialize metrics stage.

        Args:
            cfg: Full configuration dict
            preprocessed: Output from Stage 1
        """
        self.cfg = cfg
        self.preprocessed = preprocessed
        self.registry = get_metric_registry()
        self.computer = MetricComputer(self.registry)

    @classmethod
    def run(cls, cfg: dict, preprocessed: PreprocessedData) -> pd.DataFrame:
        """
        Execute Stage 2 metric computation.

        Args:
            cfg: Configuration dict
            preprocessed: Preprocessed data from Stage 1

        Returns:
            DataFrame with age and metric columns
        """
        stage = cls(cfg, preprocessed)
        return stage.execute()

    @classmethod
    def list_available_metrics(cls) -> List[str]:
        """
        List all available metrics in the registry.

        Returns:
            List of metric names
        """
        registry = get_metric_registry()
        return list(registry.get_all().keys())

    @classmethod
    def get_enabled_metrics(cls, cfg: dict) -> List[str]:
        """
        Get list of enabled metric names from config.

        Args:
            cfg: Configuration dict

        Returns:
            List of enabled metric names
        """
        return [m for m, enabled in cfg['metrics'].items() if enabled]

    def execute(self) -> pd.DataFrame:
        """
        Execute metric computation for all subjects.

        Returns:
            DataFrame with results
        """
        logger.info("="*60)
        logger.info("  Stage 2: Metric Computation")
        logger.info("="*60)

        # Get enabled metrics
        enabled_metrics = self.get_enabled_metrics(self.cfg)
        logger.info(f"[Stage2] Enabled metrics ({len(enabled_metrics)}): "
                   f"{', '.join(enabled_metrics)}")

        # Compute metrics for all subjects
        results_list = self._compute_all_subjects()

        # Build DataFrame
        df = self._build_dataframe(results_list)

        # Print summary
        self._print_summary(df, enabled_metrics)

        logger.info(f"[Stage2] Complete: {len(df)} subjects, "
                   f"{len(enabled_metrics)} metrics")

        return df

    def _compute_all_subjects(self) -> List[Dict[str, float]]:
        """
        Compute metrics for all subjects (with optional parallelization).

        Returns:
            List of result dicts
        """
        n_subj = self.preprocessed.n_subjects
        use_parallel = self.cfg['compute']['parallel']

        logger.info(f"[Stage2] Computing metrics for {n_subj} subjects "
                   f"(parallel={use_parallel})...")

        if use_parallel:
            n_jobs = self.cfg['compute']['n_jobs']
            results_list = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(self._compute_one_subject)(i)
                for i in tqdm(range(n_subj), desc="Computing metrics")
            )
        else:
            results_list = []
            for i in tqdm(range(n_subj), desc="Computing metrics"):
                results_list.append(self._compute_one_subject(i))

        return results_list

    def _compute_one_subject(self, idx: int) -> Dict[str, float]:
        """
        Compute all enabled metrics for a single subject.

        Args:
            idx: Subject index

        Returns:
            Dict of metric_name → value
        """
        try:
            # Get preprocessed features for this subject
            adj = self.preprocessed.conn_binary[idx]
            mdset = self.preprocessed.mdsets[idx]
            top_nodes = self.preprocessed.top_nodes[idx]

            # Compute metrics using direct method (bypassing thresholding)
            result = self._compute_metrics_direct(adj, mdset, top_nodes)

            return result

        except Exception as e:
            logger.error(f"Subject {idx} failed: {e}")
            enabled = self.get_enabled_metrics(self.cfg)
            return {m: np.nan for m in enabled}

    def _compute_metrics_direct(self, adj: np.ndarray, mdset: np.ndarray,
                               top_nodes: np.ndarray) -> Dict[str, float]:
        """
        Compute metrics directly from preprocessed features.

        Args:
            adj: Binary adjacency matrix
            mdset: MDSet node indices
            top_nodes: High-degree node indices

        Returns:
            Dict of metric_name → value
        """
        metric_flags = self.cfg['metrics']
        result = {}

        # Check for PC metrics (need special handling)
        need_pc = any(metric_flags.get(m, False)
                     for m in ['DC_PC', 'OCA_P', 'OCA_C', 'Prov_ratio'])
        pc_cache = None

        if need_pc:
            pc_cache = self.computer._compute_pc_cache(adj, mdset, self.cfg)

        # Compute each enabled metric
        for metric_name, enabled in metric_flags.items():
            if not enabled:
                continue

            metric = self.registry.get(metric_name)
            if metric is None:
                logger.warning(f"Metric '{metric_name}' enabled but not found")
                continue

            try:
                # Use cached PC results if available
                if metric_name in ['DC_PC', 'OCA_P', 'OCA_C', 'Prov_ratio'] and pc_cache:
                    result[metric_name] = pc_cache[metric_name]
                else:
                    result[metric_name] = metric.compute(adj, mdset, top_nodes, self.cfg)
            except Exception as e:
                logger.error(f"Failed to compute {metric_name}: {e}")
                result[metric_name] = np.nan

        return result

    def _build_dataframe(self, results_list: List[Dict[str, float]]) -> pd.DataFrame:
        """
        Build results DataFrame with age and metric columns.

        Args:
            results_list: List of metric result dicts

        Returns:
            DataFrame
        """
        logger.info("[Stage2] Building DataFrame...")

        df = pd.DataFrame(results_list)

        # Add age columns
        age = self.preprocessed.age
        if self.cfg['age']['unit'] == 'month':
            df.insert(0, 'age_months', age)
            df.insert(1, 'age_years', age / 12.0)
        else:
            df.insert(0, 'age_years', age)
            df.insert(1, 'age_months', age * 12.0)

        return df

    def _print_summary(self, df: pd.DataFrame, metric_names: List[str]):
        """
        Print summary statistics for computed metrics.

        Args:
            df: Results DataFrame
            metric_names: List of metric names
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"  Summary: {len(df)} subjects, "
                   f"age {df['age_years'].min():.1f}~{df['age_years'].max():.1f} years")
        logger.info(f"{'='*50}")

        for mname in metric_names:
            if mname not in df.columns:
                continue

            v = df[mname].replace([np.inf, -np.inf], np.nan).dropna()
            if len(v) == 0:
                logger.info(f"  {mname:12s}: No valid values")
                continue

            logger.info(f"  {mname:12s}: mean={v.mean():.4f}  std={v.std():.4f}  "
                       f"median={v.median():.4f}  (n={len(v)})")


# ================================================================
# Standalone Function (Backward Compatibility)
# ================================================================

def run_metrics(cfg: dict, preprocessed: PreprocessedData) -> pd.DataFrame:
    """
    Execute Stage 2 metric computation (backward compatible).

    Args:
        cfg: Configuration dict
        preprocessed: Preprocessed data from Stage 1

    Returns:
        DataFrame with metrics
    """
    return Stage2Metrics.run(cfg, preprocessed)


def list_available_metrics() -> List[str]:
    """List all available metrics."""
    return Stage2Metrics.list_available_metrics()
