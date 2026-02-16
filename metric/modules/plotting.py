"""
Visualization module: scatter trends, boxplots, comparisons, heatmaps.

Class-based plotting framework for brain connectivity metrics.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
from .metrics import METRIC_REGISTRY
from .utils import ensure_dir

logger = logging.getLogger(__name__)

# Global style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 100,
})


# ================================================================
# Base Plotter Class
# ================================================================

class BasePlotter:
    """
    Base class for all plotters.

    Handles common operations like saving, labels, reference lines.
    """

    def __init__(self, cfg: dict, save_dir: str, param_str: str):
        """
        Initialize plotter.

        Args:
            cfg: Configuration dict with viz settings
            save_dir: Directory to save figures
            param_str: Parameter string for filenames
        """
        self.cfg = cfg
        self.save_dir = save_dir
        self.param_str = param_str
        ensure_dir(save_dir)

    def save(self, fig, fname: str):
        """
        Save figure in requested formats.

        Args:
            fig: Matplotlib figure
            fname: Filename (without extension)
        """
        if self.cfg['viz'].get('save_png', True):
            path = os.path.join(self.save_dir, f"{fname}.png")
            fig.savefig(path, dpi=self.cfg['viz']['dpi'], bbox_inches='tight')
        plt.close(fig)

    def get_label(self, mname: str) -> str:
        """Get display label for metric."""
        reg = METRIC_REGISTRY.get(mname, {})
        return reg.get('label', mname)

    def get_ref(self, mname: str):
        """Get reference line value and text for metric."""
        reg = METRIC_REGISTRY.get(mname, {})
        return reg.get('ref_line'), reg.get('ref_text', '')


# ================================================================
# Scatter Trend Plotter
# ================================================================

class ScatterTrendPlotter(BasePlotter):
    """
    Age vs Metric scatter plot with regression line.
    """

    def plot(self, df: pd.DataFrame, mname: str, trend_row: dict):
        """
        Create scatter trend plot.

        Args:
            df: DataFrame with age and metric data
            mname: Metric name
            trend_row: Dict with regression results (slope, intercept, r, p)
        """
        vals = df[mname].values
        age = df['age_years'].values
        valid = np.isfinite(vals) & ~np.isinf(vals)

        if valid.sum() < 3:
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(age[valid], vals[valid], s=20,
                  alpha=self.cfg['viz']['scatter_alpha'],
                  c='#3366aa', edgecolors='none')

        # Regression line
        x_fit = np.linspace(age.min(), age.max(), 200)
        y_fit = trend_row['slope'] * x_fit + trend_row['intercept']
        ax.plot(x_fit, y_fit, 'r-', linewidth=2)

        # Reference line
        ref_val, ref_text = self.get_ref(mname)
        if ref_val is not None:
            ax.axhline(ref_val, ls='--', color='k', lw=1.2, label=ref_text)
            ax.legend(fontsize=9)

        # Labels and title
        ax.set_xlabel('Age (years)')
        ax.set_ylabel(self.get_label(mname))
        title = (f"Age vs {mname}  "
                f"(r={trend_row['r']:.3f}, p={trend_row['p_pearson']:.2e})  "
                f"[{self.param_str}]")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Save
        self.save(fig, f"trend_{mname}_{self.param_str}")


# ================================================================
# Boxplot Plotter
# ================================================================

class BoxplotBinsPlotter(BasePlotter):
    """
    Boxplot of metric across age bins with significance markers.
    """

    def plot(self, df: pd.DataFrame, mname: str, stat_res: Optional[dict]):
        """
        Create boxplot by age bins.

        Args:
            df: DataFrame with bin assignments and metrics
            mname: Metric name
            stat_res: Statistical test results (optional)
        """
        vals = df[mname].values
        bins = df['age_bin'].values
        valid = np.isfinite(vals) & ~np.isinf(vals) & (bins > 0)

        if valid.sum() < 10:
            return

        plot_df = df.loc[valid, ['age_bin_label', mname]].copy()

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 5.5))
        order = sorted(df.loc[valid, 'age_bin_label'].unique(),
                      key=lambda x: df.loc[df['age_bin_label'] == x, 'age_bin'].iloc[0])

        sns.boxplot(data=plot_df, x='age_bin_label', y=mname, order=order,
                   ax=ax, width=0.6, palette='Blues_d')

        # Reference line
        ref_val, ref_text = self.get_ref(mname)
        if ref_val is not None:
            ax.axhline(ref_val, ls='--', color='r', lw=1.2, label=ref_text)
            ax.legend(fontsize=9)

        # Significance brackets
        if stat_res and 'pairwise' in stat_res:
            self._add_sig_brackets(ax, vals, valid, bins, stat_res)

        # Labels and title
        kw_str = f"  KW p={stat_res['kw_p']:.2e}" if stat_res else ""
        ax.set_xlabel('Age Bin')
        ax.set_ylabel(self.get_label(mname))
        ax.set_title(f"{mname} by Age Bin{kw_str}  [{self.param_str}]")
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=20)

        # Save
        self.save(fig, f"bins_{mname}_{self.param_str}")

    def _add_sig_brackets(self, ax, vals, valid, bins, stat_res):
        """Add significance brackets to boxplot."""
        sig_pairs = [pw for pw in stat_res['pairwise'] if pw.get('sig', False)]
        if not sig_pairs:
            return

        ymax = vals[valid].max()
        yrange = np.ptp(vals[valid])
        offset = yrange * 0.06

        unique_bins = sorted(set(bins[valid]))
        bin_to_x = {b: i for i, b in enumerate(unique_bins)}

        for k, pw in enumerate(sig_pairs[:5]):  # Top 5 only
            y_line = ymax + offset * (k + 1)
            x1 = bin_to_x.get(pw['bin1'], 0)
            x2 = bin_to_x.get(pw['bin2'], 0)
            p_c = pw['p_corrected']

            star = '***' if p_c < 0.001 else '**' if p_c < 0.01 else '*'
            ax.plot([x1, x2], [y_line, y_line], 'k-', lw=1)
            ax.text((x1 + x2) / 2, y_line + offset * 0.2, star,
                   ha='center', fontsize=10)


# ================================================================
# Metric Comparison Plotter
# ================================================================

class MetricComparisonPlotter(BasePlotter):
    """
    Two-metric scatter plot colored by age.
    """

    def plot(self, df: pd.DataFrame, m1: str, m2: str):
        """
        Create metric vs metric scatter plot.

        Args:
            df: DataFrame with metrics
            m1: First metric name
            m2: Second metric name
        """
        v1 = df[m1].values
        v2 = df[m2].values
        age = df['age_years'].values
        valid = np.isfinite(v1) & ~np.isinf(v1) & np.isfinite(v2) & ~np.isinf(v2)

        if valid.sum() < 10:
            return

        from scipy.stats import pearsonr
        r, p = pearsonr(v1[valid], v2[valid])

        # Create figure
        fig, ax = plt.subplots(figsize=(7, 5.5))
        sc = ax.scatter(v1[valid], v2[valid], c=age[valid], s=25,
                       alpha=0.6, cmap=self.cfg['viz']['colormap'],
                       edgecolors='none')
        plt.colorbar(sc, ax=ax, label='Age (years)')

        # Reference lines
        for mname, axis_fn in [(m1, ax.axvline), (m2, ax.axhline)]:
            ref_val, _ = self.get_ref(mname)
            if ref_val is not None:
                axis_fn(ref_val, ls='--', color='k', lw=1, alpha=0.6)

        # Labels and title
        ax.set_xlabel(self.get_label(m1))
        ax.set_ylabel(self.get_label(m2))
        ax.set_title(f"{m1} vs {m2}  (r={r:.3f}, p={p:.2e})  [{self.param_str}]")
        ax.grid(True, alpha=0.3)

        # Save
        self.save(fig, f"comp_{m1}_vs_{m2}_{self.param_str}")


# ================================================================
# Heatmap Plotter
# ================================================================

class HeatmapSummaryPlotter(BasePlotter):
    """
    Heatmap: rows=metrics, cols=bins, values=z-scored mean.
    """

    def plot(self, bin_stats_df: pd.DataFrame):
        """
        Create summary heatmap.

        Args:
            bin_stats_df: DataFrame with bin statistics
        """
        if bin_stats_df.empty:
            return

        pivot = bin_stats_df.pivot(index='metric', columns='label', values='mean')

        # Z-score each row
        z = pivot.sub(pivot.mean(axis=1), axis=0).div(pivot.std(axis=1), axis=0)
        z = z.fillna(0)

        # Create figure
        fig_width = max(8, len(z.columns) * 1.2)
        fig_height = max(4, len(z) * 0.6)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        sns.heatmap(z, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                   vmin=-2, vmax=2, linewidths=0.5, ax=ax)

        ax.set_title(f"Metric Summary (z-scored)  [{self.param_str}]")
        ax.set_ylabel('')
        plt.xticks(rotation=25)

        # Save
        self.save(fig, f"heatmap_summary_{self.param_str}")


# ================================================================
# Standalone Functions (Backward Compatibility)
# ================================================================

def plot_scatter_trend(df: pd.DataFrame, mname: str, trend_row: dict,
                      cfg: dict, save_dir: str, pstr: str):
    """Age vs metric scatter + regression line (backward compatible)."""
    plotter = ScatterTrendPlotter(cfg, save_dir, pstr)
    plotter.plot(df, mname, trend_row)


def plot_boxplot_bins(df: pd.DataFrame, mname: str, stat_res: Optional[dict],
                      cfg: dict, save_dir: str, pstr: str):
    """Boxplot of metric across age bins (backward compatible)."""
    plotter = BoxplotBinsPlotter(cfg, save_dir, pstr)
    plotter.plot(df, mname, stat_res)


def plot_metric_comparison(df: pd.DataFrame, m1: str, m2: str,
                          cfg: dict, save_dir: str, pstr: str):
    """Two-metric scatter colored by age (backward compatible)."""
    plotter = MetricComparisonPlotter(cfg, save_dir, pstr)
    plotter.plot(df, m1, m2)


def plot_heatmap_summary(bin_stats_df: pd.DataFrame,
                        cfg: dict, save_dir: str, pstr: str):
    """Heatmap: rows=metrics, cols=bins (backward compatible)."""
    plotter = HeatmapSummaryPlotter(cfg, save_dir, pstr)
    plotter.plot(bin_stats_df)
