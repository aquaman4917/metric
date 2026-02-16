"""
Stage 4: Visualization
=======================
Pipeline:
  1. Scatter trend plots (age vs metric)
  2. Boxplots by age bins with significance markers
  3. Metric comparison plots
  4. Summary heatmaps
"""

import logging
import os
import pandas as pd
from typing import Dict, List, Optional

from .stage3_analysis import AnalysisResults
from .plotting import (
    ScatterTrendPlotter,
    BoxplotBinsPlotter,
    MetricComparisonPlotter,
    HeatmapSummaryPlotter,
)
from .utils import ensure_dir

logger = logging.getLogger(__name__)


# ================================================================
# Stage 4 Class
# ================================================================

class Stage4Visualization:
    """
    Stage 4: Visualization.

    Generates all plots for analysis results.
    """

    def __init__(self, cfg: dict, analysis_results: AnalysisResults,
                 fig_dir: str, param_str: str):
        """
        Initialize visualization stage.

        Args:
            cfg: Full configuration dict
            analysis_results: Output from Stage 3
            fig_dir: Directory to save figures
            param_str: Parameter string for filenames
        """
        self.cfg = cfg
        self.results = analysis_results
        self.fig_dir = fig_dir
        self.param_str = param_str
        ensure_dir(fig_dir)

    @classmethod
    def run(cls, cfg: dict, analysis_results: AnalysisResults,
           fig_dir: str, param_str: str):
        """
        Execute Stage 4 visualization.

        Args:
            cfg: Configuration dict
            analysis_results: Analysis results from Stage 3
            fig_dir: Directory for figures
            param_str: Parameter string
        """
        stage = cls(cfg, analysis_results, fig_dir, param_str)
        stage.execute()

    @classmethod
    def list_plot_types(cls) -> List[str]:
        """
        List available plot types.

        Returns:
            List of plot type names
        """
        return [
            'scatter_trend',
            'boxplot_bins',
            'metric_comparison',
            'heatmap_summary',
        ]

    def execute(self):
        """Execute all visualization tasks."""
        logger.info("="*60)
        logger.info("  Stage 4: Visualization")
        logger.info("="*60)

        # Step 1: Scatter trend plots
        self._plot_scatter_trends()

        # Step 2: Boxplots by age bins
        self._plot_boxplots()

        # Step 3: Metric comparisons
        self._plot_metric_comparisons()

        # Step 4: Summary heatmap
        self._plot_heatmap()

        logger.info(f"[Stage4] Complete: Figures saved to {self.fig_dir}")

    def _plot_scatter_trends(self):
        """Generate scatter trend plots for all metrics."""
        logger.info("[Step 1] Scatter Trend Plots")

        plotter = ScatterTrendPlotter(self.cfg, self.fig_dir, self.param_str)
        n_plots = 0

        for _, row in self.results.trend_df.iterrows():
            mname = row['metric']
            plotter.plot(self.results.df, mname, row.to_dict())
            n_plots += 1

        logger.info(f"[Step 1] Generated {n_plots} scatter plots")

    def _plot_boxplots(self):
        """Generate boxplots by age bins for all metrics."""
        logger.info("[Step 2] Boxplot by Age Bins")

        plotter = BoxplotBinsPlotter(self.cfg, self.fig_dir, self.param_str)
        metric_names = self.results.trend_df['metric'].tolist()
        n_plots = 0

        for mname in metric_names:
            stat_res = self.results.stat_results.get(mname)
            plotter.plot(self.results.df, mname, stat_res)
            n_plots += 1

        logger.info(f"[Step 2] Generated {n_plots} boxplots")

    def _plot_metric_comparisons(self):
        """Generate metric vs metric comparison plots."""
        logger.info("[Step 3] Metric Comparison Plots")

        plotter = MetricComparisonPlotter(self.cfg, self.fig_dir, self.param_str)
        metric_names = self.results.trend_df['metric'].tolist()

        # Define comparison pairs
        comparison_pairs = [
            ('DC', 'OCA'),
            ('DC', 'DC_PC'),
            ('OCA_P', 'OCA_C'),
            ('DC_PC', 'Prov_ratio'),
        ]

        n_plots = 0
        for m1, m2 in comparison_pairs:
            if m1 in metric_names and m2 in metric_names:
                plotter.plot(self.results.df, m1, m2)
                n_plots += 1

        logger.info(f"[Step 3] Generated {n_plots} comparison plots")

    def _plot_heatmap(self):
        """Generate summary heatmap."""
        logger.info("[Step 4] Summary Heatmap")

        plotter = HeatmapSummaryPlotter(self.cfg, self.fig_dir, self.param_str)
        plotter.plot(self.results.bin_stats_df)

        logger.info(f"[Step 4] Generated heatmap")


# ================================================================
# Standalone Function (Backward Compatibility)
# ================================================================

def run_visualization(cfg: dict, analysis_results: AnalysisResults,
                     fig_dir: str, param_str: str):
    """
    Execute Stage 4 visualization (backward compatible).

    Args:
        cfg: Configuration dict
        analysis_results: Analysis results from Stage 3
        fig_dir: Directory for figures
        param_str: Parameter string
    """
    Stage4Visualization.run(cfg, analysis_results, fig_dir, param_str)


def list_plot_types() -> List[str]:
    """List available plot types."""
    return Stage4Visualization.list_plot_types()
