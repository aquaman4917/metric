"""
Stage 3: Statistical Analysis
==============================
Pipeline:
  1. Lifespan trend analysis (correlations & regressions)
  2. Age binning & bin-level statistics
  3. Statistical tests (Kruskal-Wallis, Mann-Whitney U)
  4. Multiple comparison correction
"""

import logging
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .analysis import LifespanAnalyzer, AgeBinAnalyzer, StatisticalTestAnalyzer, QCAnalyzer

logger = logging.getLogger(__name__)


# ================================================================
# Data Structures
# ================================================================

@dataclass
class AnalysisResults:
    """
    Container for Stage 3 analysis results.

    Attributes:
        df: DataFrame with age bins assigned
        trend_df: Lifespan trend analysis results
        bin_stats_df: Per-bin summary statistics
        stat_results: Statistical test results
        qc_meta: Subject-level QC metadata
    """
    df: pd.DataFrame
    trend_df: pd.DataFrame
    bin_stats_df: pd.DataFrame
    stat_results: Dict[str, dict]
    qc_meta: Optional[Dict[str, object]] = None

    def __repr__(self) -> str:
        n_metrics = len(self.trend_df)
        n_bins = self.df['age_bin'].nunique() - 1  # Exclude 0
        return (f"AnalysisResults(n_metrics={n_metrics}, "
                f"n_bins={n_bins}, n_subjects={len(self.df)})")


# ================================================================
# Stage 3 Class
# ================================================================

class Stage3Analysis:
    """
    Stage 3: Statistical analysis.

    Performs lifespan trend analysis, age binning, and statistical tests.
    """

    def __init__(self, cfg: dict, df: pd.DataFrame, metric_names: List[str]):
        """
        Initialize analysis stage.

        Args:
            cfg: Full configuration dict
            df: DataFrame with metrics from Stage 2
            metric_names: List of metric names to analyze
        """
        self.cfg = cfg
        self.df = df
        self.metric_names = metric_names

    @classmethod
    def run(cls, cfg: dict, df: pd.DataFrame, metric_names: List[str]) -> AnalysisResults:
        """
        Execute Stage 3 analysis.

        Args:
            cfg: Configuration dict
            df: DataFrame with metrics
            metric_names: List of metric names

        Returns:
            AnalysisResults container
        """
        stage = cls(cfg, df, metric_names)
        return stage.execute()

    def execute(self) -> AnalysisResults:
        """
        Execute full analysis pipeline.

        Returns:
            AnalysisResults container
        """
        logger.info("="*60)
        logger.info("  Stage 3: Statistical Analysis")
        logger.info("="*60)

        # Step 1: QC
        df_qc, qc_meta = self._run_qc()

        # Step 2: Lifespan trend analysis
        trend_df = self._lifespan_trend_analysis(df_qc)

        # Step 3: Age binning
        df_with_bins = self._age_binning(df_qc)

        # Step 4: Bin-level statistics
        bin_stats_df = self._bin_statistics(df_with_bins)

        # Step 5: Statistical tests
        stat_results = self._statistical_tests(df_with_bins)

        # Create container
        results = AnalysisResults(
            df=df_with_bins,
            trend_df=trend_df,
            bin_stats_df=bin_stats_df,
            stat_results=stat_results,
            qc_meta=qc_meta,
        )

        logger.info(f"[Stage3] Complete: {results}")
        return results

    def _run_qc(self) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """
        Run subject-level QC on metrics.

        Returns:
            (filtered_df, qc_meta)
        """
        logger.info("[Step 1] Subject QC")
        analyzer = QCAnalyzer(self.df, self.metric_names, self.cfg)
        df_qc, qc_meta = analyzer.run()
        if df_qc.empty:
            raise ValueError("No subjects left after QC. Relax qc settings in config.")
        logger.info(f"[Step 1] Retained {len(df_qc)}/{len(self.df)} subjects after QC")
        return df_qc, qc_meta

    def _lifespan_trend_analysis(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Perform lifespan trend analysis.

        Returns:
            DataFrame with trend results
        """
        logger.info("[Step 2] Lifespan Trend Analysis")

        analyzer = LifespanAnalyzer(df_input, self.metric_names, cfg=self.cfg)
        trend_df = analyzer.analyze()

        logger.info(f"[Step 2] Analyzed {len(trend_df)} metrics")

        return trend_df

    def _age_binning(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Assign age bins to subjects.

        Returns:
            DataFrame with bin assignments
        """
        logger.info("[Step 3] Age Binning")

        analyzer = AgeBinAnalyzer(df_input, self.cfg)
        df_with_bins = analyzer.assign_bins()

        n_bins = df_with_bins['age_bin'].nunique() - 1  # Exclude 0
        logger.info(f"[Step 3] Assigned {n_bins} age bins")

        return df_with_bins

    def _bin_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-bin summary statistics.

        Args:
            df: DataFrame with bin assignments

        Returns:
            DataFrame with bin statistics
        """
        logger.info("[Step 4] Bin-Level Statistics")

        analyzer = AgeBinAnalyzer(df, self.cfg)
        bin_stats_df = analyzer.compute_bin_stats(df, self.metric_names)

        logger.info(f"[Step 4] Computed stats for {len(bin_stats_df)} bin-metric combinations")

        return bin_stats_df

    def _statistical_tests(self, df: pd.DataFrame) -> Dict[str, dict]:
        """
        Perform statistical tests across age bins.

        Args:
            df: DataFrame with bin assignments

        Returns:
            Dict of test results per metric
        """
        logger.info("[Step 5] Statistical Tests")

        analyzer = StatisticalTestAnalyzer(df, self.metric_names, self.cfg)
        stat_results = analyzer.run_tests()

        n_tested = len(stat_results)
        logger.info(f"[Step 5] Tested {n_tested} metrics")

        return stat_results

    def save_results(self, results: AnalysisResults, out_dir: str, param_str: str):
        """
        Save analysis results to CSV files.

        Args:
            results: Analysis results container
            out_dir: Output directory
            param_str: Parameter string for filenames
        """
        logger.info("[Stage3] Saving results...")

        # Save trend results
        trend_path = os.path.join(out_dir, f"trend_{param_str}.csv")
        results.trend_df.to_csv(trend_path, index=False)
        logger.info(f"  Saved: {trend_path}")

        # Save QC metadata
        if results.qc_meta is not None:
            qc_path = os.path.join(out_dir, f"qc_{param_str}.json")
            with open(qc_path, 'w', encoding='utf-8') as f:
                json.dump(results.qc_meta, f, indent=2)
            logger.info(f"  Saved: {qc_path}")

        # Save bin statistics
        bin_stats_path = os.path.join(out_dir, f"bin_stats_{param_str}.csv")
        results.bin_stats_df.to_csv(bin_stats_path, index=False)
        logger.info(f"  Saved: {bin_stats_path}")

        # Save pairwise stats
        if results.stat_results:
            stat_rows = []
            for mname, res in results.stat_results.items():
                for pw in res['pairwise']:
                    stat_rows.append({'metric': mname, **pw})

            if stat_rows:
                stat_path = os.path.join(out_dir, f"pairwise_stats_{param_str}.csv")
                pd.DataFrame(stat_rows).to_csv(stat_path, index=False)
                logger.info(f"  Saved: {stat_path}")


# ================================================================
# Standalone Function (Backward Compatibility)
# ================================================================

def run_analysis(cfg: dict, df: pd.DataFrame, metric_names: List[str]) -> AnalysisResults:
    """
    Execute Stage 3 analysis (backward compatible).

    Args:
        cfg: Configuration dict
        df: DataFrame with metrics
        metric_names: List of metric names

    Returns:
        AnalysisResults container
    """
    return Stage3Analysis.run(cfg, df, metric_names)
