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
from typing import Any, Dict, List, Optional
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


def _safe_p(pval: float) -> float:
    """Clamp p-values to finite positive range for -log10 transform."""
    if pval is None or not np.isfinite(pval):
        return np.nan
    return max(float(pval), 1e-300)


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
    Age vs Metric scatter plot with regression line (linear or quadratic).
    """

    def plot(self, df: pd.DataFrame, mname: str, trend_row: dict):
        """
        Create scatter trend plot.

        Args:
            df: DataFrame with age and metric data
            mname: Metric name
            trend_row: Dict with regression results (slope, intercept, r, p,
                       and optionally quad_a/b/c, quad_r2, best_model)
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

        x_fit = np.linspace(age[valid].min(), age[valid].max(), 300)
        best_model = trend_row.get('best_model', 'linear')

        # Always draw linear fit (thin, grey)
        y_lin = trend_row['slope'] * x_fit + trend_row['intercept']
        ax.plot(x_fit, y_lin, color='#999999', linewidth=1.2, alpha=0.6,
                linestyle='--', label=f"linear (R²={trend_row.get('linear_r2', 0):.3f})")

        # Quadratic fit (bold red) if available and selected
        if best_model == 'quad' and 'quad_a' in trend_row:
            a, b, c = trend_row['quad_a'], trend_row['quad_b'], trend_row['quad_c']
            y_quad = a * x_fit**2 + b * x_fit + c
            ax.plot(x_fit, y_quad, 'r-', linewidth=2.2,
                    label=f"quadratic (R²={trend_row.get('quad_r2', 0):.3f})")
            # Mark vertex (peak/trough of inverted-U)
            vertex_x = -b / (2 * a) if abs(a) > 1e-15 else None
            if vertex_x is not None and age[valid].min() <= vertex_x <= age[valid].max():
                vertex_y = a * vertex_x**2 + b * vertex_x + c
                ax.plot(vertex_x, vertex_y, 'r*', ms=12, zorder=5)
                ax.annotate(f'peak ≈ {vertex_x:.0f}y',
                           (vertex_x, vertex_y),
                           textcoords='offset points', xytext=(10, 10),
                           fontsize=9, color='red',
                           arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
        elif best_model == 'cubic' and 'cubic_coeffs' in trend_row:
            coeffs = trend_row['cubic_coeffs']
            y_cub = np.polyval(coeffs, x_fit)
            ax.plot(x_fit, y_cub, 'r-', linewidth=2.2,
                    label=f"cubic (R²={trend_row.get('cubic_r2', 0):.3f})")
        else:
            # Linear is best → make it bold red instead of grey
            ax.lines[-1].set_color('red')
            ax.lines[-1].set_linewidth(2)
            ax.lines[-1].set_alpha(1.0)
            ax.lines[-1].set_linestyle('-')

        ax.legend(fontsize=8, loc='best')

        # Reference line
        ref_val, ref_text = self.get_ref(mname)
        if ref_val is not None:
            ax.axhline(ref_val, ls='--', color='k', lw=1.2, label=ref_text)

        # Labels and title
        ax.set_xlabel('Age (years)')
        ax.set_ylabel(self.get_label(mname))

        # Build informative title
        r2_best = trend_row.get(f'{best_model}_r2', trend_row.get('linear_r2', 0))
        quad_info = ''
        if best_model == 'quad' and 'quad_p' in trend_row:
            quad_info = f", F-test p={trend_row['quad_p']:.2e}"
        title = (f"{mname} vs Age  "
                f"[best: {best_model}, R²={r2_best:.4f}{quad_info}]  "
                f"(Pearson r={trend_row['r']:.3f})")
        ax.set_title(title, fontsize=10)
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
# Trend + Bin Summary Plotter
# ================================================================

class TrendBinsSummaryPlotter(BasePlotter):
    """
    Summary figure:
      1) Age vs metric trend for all metrics (whole cohort)
      2) Same metrics divided by configured age bins
    """

    def _ordered_bin_labels(self, df: pd.DataFrame) -> List[str]:
        if 'age_bin_label' not in df.columns or 'age_bin' not in df.columns:
            return []
        uniq = df[['age_bin_label', 'age_bin']].dropna().drop_duplicates().copy()
        uniq['age_bin_num'] = pd.to_numeric(uniq['age_bin'], errors='coerce')
        uniq = uniq[np.isfinite(uniq['age_bin_num']) & (uniq['age_bin_num'] > 0)]
        uniq = uniq.sort_values('age_bin_num')
        return uniq['age_bin_label'].tolist()

    def _smooth_trend(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return robust smooth trend line for scatter."""
        if len(x) < 4:
            order = np.argsort(x)
            return x[order], y[order]

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        # Robustly remove extreme y outliers for trend fitting only.
        # (Scatter still shows all points.)
        y_med = np.nanmedian(y)
        mad = np.nanmedian(np.abs(y - y_med)) + 1e-10
        robust_z = 0.6745 * (y - y_med) / mad
        keep = np.abs(robust_z) <= 4.5
        # Fallback: if too aggressive, keep original.
        if keep.sum() >= max(8, int(0.3 * len(y))):
            x_fit = x[keep]
            y_fit = y[keep]
        else:
            x_fit = x
            y_fit = y

        # Merge duplicate ages before smoothing.
        grouped = pd.DataFrame({'x': x_fit, 'y': y_fit}).groupby('x', as_index=False).median()
        xu = grouped['x'].to_numpy(dtype=float)
        yu = grouped['y'].to_numpy(dtype=float)

        if len(xu) < 4:
            return xu, yu

        try:
            # Quantile-bin medians -> shape-preserving interpolation (no spline overshoot).
            n_bins = int(np.clip(len(xu) // 8, 8, 28))
            edges = np.quantile(xu, np.linspace(0, 1, n_bins + 1))
            xk, yk = [], []
            for i in range(n_bins):
                lo, hi = edges[i], edges[i + 1]
                if i == n_bins - 1:
                    m = (xu >= lo) & (xu <= hi)
                else:
                    m = (xu >= lo) & (xu < hi)
                if not m.any():
                    continue
                xk.append(np.median(xu[m]))
                yk.append(np.median(yu[m]))

            xk = np.asarray(xk, dtype=float)
            yk = np.asarray(yk, dtype=float)
            if len(xk) < 4:
                return xu, yu

            from scipy.interpolate import PchipInterpolator
            spline = PchipInterpolator(xk, yk, extrapolate=False)
            x_grid = np.linspace(xu.min(), xu.max(), 320)
            y_grid = spline(x_grid)
            good = np.isfinite(y_grid)
            return x_grid[good], y_grid[good]
        except Exception:
            # Fallback: centered rolling mean over unique-age profile.
            win = max(5, int(round(0.15 * len(xu))))
            if win % 2 == 0:
                win += 1
            y_roll = pd.Series(yu).rolling(window=win, center=True, min_periods=1).mean().to_numpy()
            return xu, y_roll

    def plot(self, df: pd.DataFrame, metric_names: List[str],
             trend_df: Optional[pd.DataFrame] = None):
        present = [m for m in metric_names if m in df.columns]
        if not present or 'age_years' not in df.columns:
            return

        from scipy import stats as sp_stats

        # Build trend lookup if available
        trend_map = {}
        if trend_df is not None and not trend_df.empty:
            trend_map = trend_df.set_index('metric').to_dict(orient='index')

        age = pd.to_numeric(df['age_years'], errors='coerce').to_numpy(dtype=float)
        labels = self._ordered_bin_labels(df)

        n_rows = len(present)
        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=2,
            figsize=(15, max(7, 3.4 * n_rows)),
            squeeze=False,
        )

        for i, mname in enumerate(present):
            vals = pd.to_numeric(df[mname], errors='coerce').to_numpy(dtype=float)
            valid = np.isfinite(age) & np.isfinite(vals)

            # Left: scatter + smooth trend line + polynomial fit
            ax_l = axes[i, 0]
            if valid.sum() >= 4:
                x = age[valid]
                y = vals[valid]
                ax_l.scatter(
                    x, y, s=18, alpha=self.cfg['viz'].get('scatter_alpha', 0.5),
                    c='#3366aa', edgecolors='none'
                )

                # Smooth trend (LOESS-like)
                tx, ty = self._smooth_trend(x, y)
                ax_l.plot(tx, ty, color='#d62728', lw=2.2, alpha=0.95)

                # Overlay polynomial fit if available
                tr = trend_map.get(mname, {})
                best = tr.get('best_model', 'linear')
                x_fit = np.linspace(x.min(), x.max(), 300)

                if best == 'quad' and 'quad_a' in tr:
                    a, b, c = tr['quad_a'], tr['quad_b'], tr['quad_c']
                    y_quad = a * x_fit**2 + b * x_fit + c
                    ax_l.plot(x_fit, y_quad, color='#ff7f0e', lw=1.8,
                              ls='--', alpha=0.85, label=f"quad R²={tr.get('quad_r2',0):.3f}")
                    # Mark vertex
                    vx = -b / (2*a) if abs(a) > 1e-15 else None
                    if vx is not None and x.min() <= vx <= x.max():
                        vy = a*vx**2 + b*vx + c
                        ax_l.plot(vx, vy, '*', color='#ff7f0e', ms=10, zorder=5)
                    ax_l.legend(fontsize=7, loc='best')

                try:
                    r, p = sp_stats.pearsonr(x, y)
                    r2_str = f"R²={tr.get(f'{best}_r2', r**2):.3f}"
                    st = f"{best}: {r2_str}, r={r:.3f}"
                except Exception:
                    st = "r=NA"
            else:
                st = "insufficient data"

            ref_val, ref_txt = self.get_ref(mname)
            if ref_val is not None:
                ax_l.axhline(ref_val, ls='--', color='k', lw=1.0, alpha=0.7)
                if ref_txt:
                    ax_l.text(0.98, 0.03, ref_txt, transform=ax_l.transAxes,
                              ha='right', va='bottom', fontsize=8, color='k')

            ax_l.set_title(f"{mname} vs Age ({st})", fontsize=10)
            ax_l.set_xlabel('Age (years)')
            ax_l.set_ylabel(self.get_label(mname))
            ax_l.grid(True, alpha=0.22)

            # Right: divided by bin (box + points in one panel)
            ax_r = axes[i, 1]
            if labels and ('age_bin_label' in df.columns):
                plot_df = df.loc[valid, ['age_bin_label', mname]].copy()
                sns.boxplot(
                    data=plot_df, x='age_bin_label', y=mname, order=labels,
                    width=0.55, color='#cfe2f3', fliersize=0, ax=ax_r
                )
                sns.stripplot(
                    data=plot_df, x='age_bin_label', y=mname, order=labels,
                    size=3, alpha=0.35, color='#1f4e79', jitter=0.18, ax=ax_r
                )
                # Mean trend across bins
                means = [plot_df.loc[plot_df['age_bin_label'] == lb, mname].mean() for lb in labels]
                ax_r.plot(range(len(labels)), means, color='#d62728', lw=1.8, marker='o', ms=3)
                ax_r.tick_params(axis='x', rotation=18)
                ax_r.set_title(f"{mname} by Age Bin", fontsize=10)
                ax_r.set_xlabel('Age Bin')
                ax_r.set_ylabel(self.get_label(mname))
                ax_r.grid(True, axis='y', alpha=0.22)
            else:
                ax_r.axis('off')
                ax_r.text(0.03, 0.9, "No age_bin columns available", transform=ax_r.transAxes)

        fig.suptitle(f"Summary: Trend + Bin Split [{self.param_str}]", fontsize=14, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.985])

        if self.cfg['viz'].get('save_png', True):
            fig.savefig(
                os.path.join(self.save_dir, "summary_trend_bins.png"),
                dpi=self.cfg['viz']['dpi'],
                bbox_inches='tight',
            )
            fig.savefig(
                os.path.join(self.save_dir, f"summary_trend_bins_{self.param_str}.png"),
                dpi=self.cfg['viz']['dpi'],
                bbox_inches='tight',
            )
        plt.close(fig)


# ================================================================
# QC Plotter
# ================================================================

class QCProfilePlotter(BasePlotter):
    """QC diagnostic scatter: subject metric mean vs std."""

    def plot(self, qc_meta: Optional[dict]):
        if not qc_meta:
            return

        means = np.asarray(qc_meta.get('subject_metric_mean', []), dtype=float)
        stds = np.asarray(qc_meta.get('subject_metric_std', []), dtype=float)
        inlier_mask = np.asarray(qc_meta.get('inlier_mask', []), dtype=bool)

        if len(means) == 0 or len(stds) == 0 or len(inlier_mask) != len(means):
            return

        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.scatter(means[inlier_mask], stds[inlier_mask], s=24, alpha=0.55,
                   c='#3c78a8', edgecolors='none', label='Inliers')
        if (~inlier_mask).any():
            ax.scatter(means[~inlier_mask], stds[~inlier_mask], s=38, alpha=0.9,
                       c='#cc0000', marker='x', label='Outliers')

        method = qc_meta.get('method', 'none')
        n_out = int(qc_meta.get('n_outliers', 0))
        n_total = int(qc_meta.get('n_subjects', len(means)))
        thr = qc_meta.get('zscore_threshold')
        factor = qc_meta.get('iqr_factor')
        param_txt = f"threshold={thr}" if method == 'zscore' else (
            f"factor={factor}" if method == 'iqr' else "param=NA"
        )

        ax.set_title(f"QC Summary [{self.param_str}]")
        ax.set_xlabel('Per-subject metric mean')
        ax.set_ylabel('Per-subject metric std')
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
        ax.text(
            0.02, 0.98,
            f"method={method} | outliers={n_out}/{n_total} ({(n_out/max(n_total,1)):.1%}) | {param_txt}",
            transform=ax.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='none')
        )

        self.save(fig, f"qc_summary_{self.param_str}")


# ================================================================
# Overview Plotter
# ================================================================

class OverviewStatsPlotter(BasePlotter):
    """
    Single-page overview of all enabled metrics with statistical tests.

    Included tests:
      - Pearson correlation p
      - Spearman correlation p
      - Kruskal-Wallis p
      - One-way ANOVA p
      - Min pairwise corrected p
      - Number of significant pairwise tests
    """

    def _build_table(self, metric_names: List[str], trend_df: pd.DataFrame,
                     stat_results: Dict[str, dict]) -> pd.DataFrame:
        trend_map = {}
        if trend_df is not None and not trend_df.empty and 'metric' in trend_df.columns:
            trend_map = trend_df.set_index('metric').to_dict(orient='index')

        rows = []
        for mname in metric_names:
            tr = trend_map.get(mname, {})
            sr = stat_results.get(mname, {})
            pairwise = sr.get('pairwise', []) if isinstance(sr, dict) else []
            pw_corr = [
                pw.get('p_corrected', pw.get('p'))
                for pw in pairwise
                if np.isfinite(pw.get('p_corrected', pw.get('p', np.nan)))
            ]
            row = {
                'metric': mname,
                'pearson_r': tr.get('r', np.nan),
                'pearson_p': tr.get('p_pearson', np.nan),
                'spearman_rho': tr.get('rho', np.nan),
                'spearman_p': tr.get('p_spearman', np.nan),
                'kw_p': sr.get('kw_p', np.nan) if isinstance(sr, dict) else np.nan,
                'anova_p': sr.get('anova_p', np.nan) if isinstance(sr, dict) else np.nan,
                'pairwise_min_p_corr': float(np.nanmin(pw_corr)) if pw_corr else np.nan,
                'pairwise_sig_n': int(sr.get('n_sig', 0)) if isinstance(sr, dict) else 0,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def plot(self, metric_names: List[str], trend_df: pd.DataFrame,
             stat_results: Dict[str, dict], qc_meta: Optional[dict] = None):
        if not metric_names:
            return

        table = self._build_table(metric_names, trend_df, stat_results)
        if table.empty:
            return

        # Save machine-readable summary for quick review
        table_path = os.path.join(self.save_dir, f"overview_stats_{self.param_str}.csv")
        table.to_csv(table_path, index=False)

        p_cols = ['pearson_p', 'spearman_p', 'kw_p', 'anova_p', 'pairwise_min_p_corr']
        p_mat = table[p_cols].copy()
        # pandas>=3 removed DataFrame.applymap; Series.map is stable.
        p_mat = p_mat.apply(lambda col: col.map(_safe_p))
        p_log = -np.log10(p_mat)
        p_log.index = table['metric']
        p_log.columns = ['Pearson', 'Spearman', 'KW', 'ANOVA', 'Pairwise(min corr)']

        eff = table[['metric', 'pearson_r', 'spearman_rho']].set_index('metric')
        sig_counts = table.set_index('metric')['pairwise_sig_n']

        fig = plt.figure(figsize=(16, max(9, 0.6 * len(table) + 4)))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.9, 1.0], height_ratios=[1.5, 1.0])

        # Panel A: all test p-values
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(p_log, cmap='YlOrRd', linewidths=0.4, linecolor='white',
                    cbar_kws={'label': '-log10(p)'}, ax=ax1)
        ax1.set_title('Overview: Statistical Tests (-log10 p)')
        ax1.set_xlabel('')
        ax1.set_ylabel('')

        # Panel B: effect directions
        ax2 = fig.add_subplot(gs[0, 1])
        sns.heatmap(eff, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    linewidths=0.4, linecolor='white',
                    cbar_kws={'label': 'Effect size'}, ax=ax2)
        ax2.set_title('Trend Effect Direction')
        ax2.set_xlabel('')
        ax2.set_ylabel('')

        # Panel C: significant pairwise counts
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.bar(sig_counts.index, sig_counts.values, color='#2f7ed8', alpha=0.85)
        ax3.set_title('Significant Pairwise Tests per Metric')
        ax3.set_ylabel('Count')
        ax3.grid(True, axis='y', alpha=0.25)
        ax3.tick_params(axis='x', rotation=20)

        # Panel D: QC + overview text
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        if qc_meta:
            n_out = int(qc_meta.get('n_outliers', 0))
            n_total = int(qc_meta.get('n_subjects', 0))
            method = qc_meta.get('method', 'none')
            qc_line = f"QC: method={method}, outliers={n_out}/{n_total} ({(n_out/max(n_total,1)):.1%})"
        else:
            qc_line = "QC: not available"
        strongest = table.sort_values('pairwise_sig_n', ascending=False).head(3)
        top_lines = [
            f"{r.metric}: sig_pairs={int(r.pairwise_sig_n)}, KW p={r.kw_p:.2e}"
            for r in strongest.itertuples(index=False)
        ]
        text = "Overview Notes\n" + qc_line + "\nTop metrics by pairwise significance:\n- " + "\n- ".join(top_lines)
        ax4.text(0.0, 1.0, text, va='top', ha='left', fontsize=10)

        fig.suptitle(f"Metric Statistical Overview [{self.param_str}]", fontsize=14, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        if self.cfg['viz'].get('save_png', True):
            # Consensus-style primary name
            fig.savefig(
                os.path.join(self.save_dir, "overview_clusters.png"),
                dpi=self.cfg['viz']['dpi'],
                bbox_inches='tight',
            )
            # Keep parameterized legacy name for backward compatibility
            fig.savefig(
                os.path.join(self.save_dir, f"overview_stats_{self.param_str}.png"),
                dpi=self.cfg['viz']['dpi'],
                bbox_inches='tight',
            )
        plt.close(fig)


# ================================================================
# Convenience Functions (for notebook / interactive use)
# ================================================================
#
# The canonical pipeline uses Stage4Visualization (stage4_visualization.py).
# These thin wrappers exist only for quick ad-hoc plotting in notebooks.
# They are NOT used by main.py.
# ================================================================

def plot_scatter_trend(df, mname, trend_row, cfg, save_dir, pstr):
    ScatterTrendPlotter(cfg, save_dir, pstr).plot(df, mname, trend_row)

def plot_boxplot_bins(df, mname, stat_res, cfg, save_dir, pstr):
    BoxplotBinsPlotter(cfg, save_dir, pstr).plot(df, mname, stat_res)

def plot_metric_comparison(df, m1, m2, cfg, save_dir, pstr):
    MetricComparisonPlotter(cfg, save_dir, pstr).plot(df, m1, m2)

def plot_heatmap_summary(bin_stats_df, cfg, save_dir, pstr):
    HeatmapSummaryPlotter(cfg, save_dir, pstr).plot(bin_stats_df)

def plot_qc_summary(qc_meta, cfg, save_dir, pstr):
    QCProfilePlotter(cfg, save_dir, pstr).plot(qc_meta)

def plot_summary_trend_bins(df, metric_names, cfg, save_dir, pstr):
    TrendBinsSummaryPlotter(cfg, save_dir, pstr).plot(df, metric_names)

def plot_overview_stats(metric_names, trend_df, stat_results, qc_meta, cfg, save_dir, pstr):
    OverviewStatsPlotter(cfg, save_dir, pstr).plot(metric_names, trend_df, stat_results, qc_meta=qc_meta)
