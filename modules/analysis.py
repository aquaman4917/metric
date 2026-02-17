"""
Analysis module: lifespan trends, age binning, statistical tests.

Class-based analysis framework for brain connectivity metrics.
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from typing import Any, Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


# ================================================================
# Lifespan Trend Analyzer
# ================================================================

class LifespanAnalyzer:
    """
    Analyzes age-metric correlations and lifespan trends.

    Supports linear, quadratic, and cubic trend models.
    Always reports Pearson/Spearman correlations.
    When trend_model != 'linear', also fits polynomial and reports
    R², AIC, and F-test comparing linear vs non-linear.
    """

    def __init__(self, df: pd.DataFrame, metric_names: List[str],
                 cfg: Optional[dict] = None):
        self.df = df
        self.metric_names = metric_names
        self.cfg = cfg or {}
        self.trend_model = self.cfg.get('stats', {}).get('trend_model', 'linear')

    def analyze(self) -> pd.DataFrame:
        age = self.df['age_years'].values
        rows = []

        for mname in self.metric_names:
            if mname not in self.df.columns:
                continue

            vals = self.df[mname].values
            valid = np.isfinite(vals) & ~np.isinf(vals)

            if valid.sum() < 10:
                logger.warning(f"[trend] {mname}: only {valid.sum()} valid values, skipping")
                continue

            x, y = age[valid], vals[valid]
            n = len(x)

            r, p_pear = sp_stats.pearsonr(x, y)
            rho, p_spear = sp_stats.spearmanr(x, y)

            slope, intercept = np.polyfit(x, y, 1)
            y_pred_lin = slope * x + intercept
            ss_res_lin = np.sum((y - y_pred_lin) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            linear_r2 = 1.0 - ss_res_lin / ss_tot if ss_tot > 0 else 0.0

            row = {
                'metric': mname,
                'r': r,
                'p_pearson': p_pear,
                'rho': rho,
                'p_spearman': p_spear,
                'slope': slope,
                'intercept': intercept,
                'n': n,
                'linear_r2': linear_r2,
            }

            if self.trend_model in ('quadratic', 'cubic') and n >= 15:
                row.update(self._fit_quadratic(x, y, ss_res_lin, ss_tot, n))

            if self.trend_model == 'cubic' and n >= 20:
                row.update(self._fit_cubic(x, y, ss_res_lin, ss_tot, n))

            row['best_model'] = self._select_best_model(row)
            rows.append(row)

            best = row['best_model']
            best_r2 = row.get(f"{best}_r2", linear_r2)
            logger.info(
                f"[trend] {mname}: r={r:.3f} (p={p_pear:.2e}), "
                f"best={best} (R²={best_r2:.4f})"
            )

        return pd.DataFrame(rows)

    def _fit_quadratic(self, x, y, ss_res_lin, ss_tot, n) -> dict:
        coeffs = np.polyfit(x, y, 2)
        a, b, c = coeffs
        y_pred = np.polyval(coeffs, x)
        ss_res_quad = np.sum((y - y_pred) ** 2)
        quad_r2 = 1.0 - ss_res_quad / ss_tot if ss_tot > 0 else 0.0

        df1 = 1
        df2 = n - 3
        if df2 > 0 and ss_res_quad > 0:
            f_stat = ((ss_res_lin - ss_res_quad) / df1) / (ss_res_quad / df2)
            f_p = 1.0 - sp_stats.f.cdf(f_stat, df1, df2)
        else:
            f_stat, f_p = 0.0, 1.0

        aic_lin = n * np.log(ss_res_lin / n + 1e-30) + 2 * 2
        aic_quad = n * np.log(ss_res_quad / n + 1e-30) + 2 * 3

        return {
            'quad_a': a, 'quad_b': b, 'quad_c': c,
            'quad_r2': quad_r2,
            'quad_f': f_stat, 'quad_p': f_p,
            'aic_linear': aic_lin, 'aic_quad': aic_quad,
        }

    def _fit_cubic(self, x, y, ss_res_lin, ss_tot, n) -> dict:
        coeffs = np.polyfit(x, y, 3)
        y_pred = np.polyval(coeffs, x)
        ss_res_cub = np.sum((y - y_pred) ** 2)
        cubic_r2 = 1.0 - ss_res_cub / ss_tot if ss_tot > 0 else 0.0
        aic_cub = n * np.log(ss_res_cub / n + 1e-30) + 2 * 4

        return {
            'cubic_coeffs': coeffs.tolist(),
            'cubic_r2': cubic_r2,
            'aic_cubic': aic_cub,
        }

    def _select_best_model(self, row: dict) -> str:
        candidates = {'linear': row.get('aic_linear', np.inf)}

        if 'aic_quad' in row:
            if row.get('quad_p', 1.0) < 0.05:
                candidates['quad'] = row['aic_quad']

        if 'aic_cubic' in row:
            candidates['cubic'] = row['aic_cubic']

        if not candidates or all(np.isinf(v) for v in candidates.values()):
            return 'linear'

        return min(candidates, key=candidates.get)


# ================================================================
# Partial Correlation Analyzer
# ================================================================

class PartialCorrelationAnalyzer:
    """
    Computes partial correlations between age and each metric,
    controlling for one or more covariates (default: MDS_size).

    Method: residualization (OLS).
      1. Regress metric on covariates → residuals_y
      2. Regress age    on covariates → residuals_x
      3. Pearson r(residuals_x, residuals_y)

    Equivalent to the partial correlation controlling for covariates
    on both sides. No external dependencies beyond numpy/scipy.

    Typical use — disentangle MDS_size from age effects::

        analyzer = PartialCorrelationAnalyzer(df, metric_names,
                                              covariates=['MDS_size'])
        results = analyzer.analyze()
        # compare r_zero vs r_partial to see how much MDS_size explains
    """

    def __init__(self, df: pd.DataFrame, metric_names: List[str],
                 covariates: List[str] = None, cfg: Optional[dict] = None):
        """
        Args:
            df:            DataFrame with 'age_years', metrics, and covariates.
            metric_names:  Metrics to analyze.
            covariates:    Columns to control for. Defaults to ['MDS_size'].
            cfg:           Unused; kept for API consistency.
        """
        self.df = df
        self.metric_names = metric_names
        if covariates is None:
            requested_covariates = ['MDS_size', 'MDS_size_fraction']
        else:
            requested_covariates = list(covariates)
        self.covariates = []
        for c in requested_covariates:
            if c in self.df.columns and c not in self.covariates:
                self.covariates.append(c)
        missing_covariates = [c for c in requested_covariates if c not in self.df.columns]
        if missing_covariates:
            logger.warning(
                "[partial_corr] Missing covariates skipped: %s",
                ", ".join(missing_covariates),
            )
        self.cfg = cfg or {}

    def analyze(self) -> pd.DataFrame:
        """
        Run partial correlations for all metrics.

        Returns:
            DataFrame with columns:
                metric, r_partial, p_partial, r_zero, p_zero, n, covariates
        """
        out_cols = ['metric', 'r_partial', 'p_partial', 'r_zero', 'p_zero', 'n', 'covariates']
        if not self.covariates:
            logger.warning("[partial_corr] No valid covariates found. Analysis skipped.")
            return pd.DataFrame(columns=out_cols)

        rows = []
        for mname in self.metric_names:
            if mname not in self.df.columns:
                continue
            if mname in self.covariates:
                logger.info(f"[partial_corr] {mname}: metric is covariate, skipping")
                continue
            result = self._partial_corr_one(mname)
            if result is not None:
                rows.append(result)
        return pd.DataFrame(rows, columns=out_cols)

    def _partial_corr_one(self, mname: str) -> Optional[dict]:
        cols_needed = ['age_years', mname] + self.covariates
        sub = self.df[cols_needed].copy()
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna()

        n = len(sub)
        if n < 20:
            logger.warning(f"[partial_corr] {mname}: only {n} valid rows, skipping")
            return None

        age = sub['age_years'].values
        y   = sub[mname].values
        X   = sub[self.covariates].values

        r_zero, p_zero = sp_stats.pearsonr(age, y)

        resid_age    = self._ols_residuals(X, age)
        resid_metric = self._ols_residuals(X, y)

        r_partial, p_partial = sp_stats.pearsonr(resid_age, resid_metric)

        logger.info(
            f"[partial_corr] {mname}: r_zero={r_zero:.3f} -> "
            f"r_partial={r_partial:.3f} (p={p_partial:.2e}), "
            f"covariates={self.covariates}"
        )

        return {
            'metric':     mname,
            'r_partial':  r_partial,
            'p_partial':  p_partial,
            'r_zero':     r_zero,
            'p_zero':     p_zero,
            'n':          n,
            'covariates': '+'.join(self.covariates),
        }

    @staticmethod
    def _ols_residuals(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """OLS residuals of y ~ intercept + X."""
        X_aug = np.column_stack([np.ones(len(y)), X])
        beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        return y - X_aug @ beta


# ================================================================
# Age Bin Analyzer
# ================================================================

class AgeBinAnalyzer:
    """
    Handles age binning and bin-level statistics.
    """

    def __init__(self, df: pd.DataFrame, cfg: dict):
        self.df = df
        self.cfg = cfg

    def assign_bins(self) -> pd.DataFrame:
        df = self.df.copy()
        age_m = df['age_months'].values
        bins_cfg = self.cfg['age'].get('bins')

        if bins_cfg is not None and len(bins_cfg) > 0:
            bin_idx, labels = self._manual_binning(age_m, bins_cfg)
        else:
            auto = self.cfg['age']['auto_bins']
            bin_idx, labels = self._auto_binning(age_m, auto)

        df['age_bin'] = bin_idx
        label_map = {i + 1: labels[i] for i in range(len(labels))}
        label_map[0] = 'unassigned'
        df['age_bin_label'] = df['age_bin'].map(label_map)

        unassigned = (bin_idx == 0).sum()
        if unassigned > 0:
            logger.warning(f"[bins] {unassigned} subjects not assigned to any bin")

        for bi in sorted(set(bin_idx) - {0}):
            n = (bin_idx == bi).sum()
            logger.info(f"[bins] Bin {bi} ({label_map[bi]}): n={n}")

        return df

    def _manual_binning(self, age_m, bins_cfg):
        bin_idx = np.zeros(len(age_m), dtype=np.int32)
        labels = []
        for i, b in enumerate(bins_cfg):
            low, high, label = b[0], b[1], b[2]
            labels.append(label)
            mask = (age_m >= low) & (age_m < high)
            bin_idx[mask] = i + 1
        return bin_idx, labels

    def _auto_binning(self, age_m, auto_cfg):
        n_bins = auto_cfg['count']
        method = auto_cfg['method']

        if method == 'quantile':
            quantiles = np.linspace(0, 100, n_bins + 1)
            edges = np.percentile(age_m, quantiles)
        else:
            edges = np.linspace(age_m.min(), age_m.max() + 1, n_bins + 1)

        bin_idx = np.digitize(age_m, edges[1:-1]) + 1
        labels = [f"{edges[i]:.0f}-{edges[i+1]:.0f}m" for i in range(n_bins)]
        return bin_idx, labels

    def compute_bin_stats(self, df: pd.DataFrame,
                          metric_names: List[str]) -> pd.DataFrame:
        rows = []
        bins = df['age_bin'].values
        unique_bins = sorted(set(bins) - {0})

        for mname in metric_names:
            if mname not in df.columns:
                continue
            vals = df[mname].values

            for bi in unique_bins:
                mask = (bins == bi) & np.isfinite(vals) & ~np.isinf(vals)
                v = vals[mask]
                if len(v) == 0:
                    continue

                rows.append({
                    'metric': mname,
                    'bin': bi,
                    'label': df.loc[mask, 'age_bin_label'].iloc[0],
                    'n': len(v),
                    'mean': v.mean(),
                    'std': v.std(),
                    'median': np.median(v),
                    'iqr': sp_stats.iqr(v),
                })

        return pd.DataFrame(rows)


# ================================================================
# Statistical Test Analyzer
# ================================================================

class StatisticalTestAnalyzer:
    """
    Performs statistical tests across age bins.
    """

    def __init__(self, df: pd.DataFrame, metric_names: List[str], cfg: dict):
        self.df = df
        self.metric_names = metric_names
        self.cfg = cfg

    def run_tests(self) -> Dict[str, dict]:
        alpha = self.cfg['stats']['alpha']
        correction = self.cfg['stats']['correction']
        bins = self.df['age_bin'].values
        unique_bins = sorted(set(bins) - {0})
        results = {}

        for mname in self.metric_names:
            if mname not in self.df.columns:
                continue

            vals = self.df[mname].values
            valid = np.isfinite(vals) & ~np.isinf(vals) & (bins > 0)

            if valid.sum() < 20:
                continue

            groups = [vals[(bins == bi) & valid] for bi in unique_bins]
            groups = [g for g in groups if len(g) >= 3]

            if len(groups) < 2:
                continue

            kw_stat, kw_p = sp_stats.kruskal(*groups)
            try:
                anova_f, anova_p = sp_stats.f_oneway(*groups)
            except Exception:
                anova_f, anova_p = np.nan, np.nan

            pairwise = self._pairwise_tests(vals, bins, unique_bins, valid)

            if pairwise:
                raw_pvals = [pw['p'] for pw in pairwise]
                corrected = self._correct_pvalues(np.array(raw_pvals), correction)
                for k, pw in enumerate(pairwise):
                    pw['p_corrected'] = corrected[k]
                    pw['sig'] = corrected[k] < alpha

            n_sig = sum(1 for pw in pairwise if pw.get('sig', False))
            results[mname] = {
                'kw_stat': kw_stat,
                'kw_p': kw_p,
                'anova_f': anova_f,
                'anova_p': anova_p,
                'pairwise': pairwise,
                'n_sig': n_sig,
            }

            logger.info(
                f"[stat] {mname}: KW p={kw_p:.2e}, ANOVA p={anova_p:.2e}, "
                f"{n_sig}/{len(pairwise)} pairwise sig ({correction})"
            )

        return results

    def _pairwise_tests(self, vals, bins, unique_bins, valid):
        pairwise = []
        for i in range(len(unique_bins)):
            for j in range(i + 1, len(unique_bins)):
                g1 = vals[(bins == unique_bins[i]) & valid]
                g2 = vals[(bins == unique_bins[j]) & valid]

                if len(g1) < 3 or len(g2) < 3:
                    continue

                u_stat, p_val = sp_stats.mannwhitneyu(g1, g2, alternative='two-sided')

                n_total = len(g1) + len(g2)
                mu = len(g1) * len(g2) / 2
                sigma = np.sqrt(len(g1) * len(g2) * (n_total + 1) / 12)
                z = (u_stat - mu) / sigma if sigma > 0 else 0
                effect_r = abs(z) / np.sqrt(n_total)

                pairwise.append({
                    'bin1': unique_bins[i],
                    'bin2': unique_bins[j],
                    'p': p_val,
                    'z': z,
                    'effect_r': effect_r,
                })
        return pairwise

    def _correct_pvalues(self, pvals, method):
        if method == 'bonferroni':
            return np.minimum(pvals * len(pvals), 1.0)
        elif method == 'fdr':
            return self._fdr_bh(pvals)
        else:
            return pvals

    def _fdr_bh(self, pvals):
        m = len(pvals)
        sorted_idx = np.argsort(pvals)
        sorted_p = pvals[sorted_idx]
        adjusted = sorted_p * m / np.arange(1, m + 1)
        adjusted = np.minimum(adjusted, 1.0)
        for i in range(m - 2, -1, -1):
            adjusted[i] = min(adjusted[i], adjusted[i + 1])
        result = np.empty(m)
        result[sorted_idx] = adjusted
        return result


# ================================================================
# QC Analyzer
# ================================================================

class QCAnalyzer:
    """
    Subject-level QC using metric profiles.
    """

    def __init__(self, df: pd.DataFrame, metric_names: List[str], cfg: dict):
        self.df = df
        self.metric_names = metric_names
        self.cfg = cfg

    def run(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        qc_cfg = self.cfg.get('qc', {})
        method = str(qc_cfg.get('method', 'none')).lower()
        z_thr = float(qc_cfg.get('zscore_threshold', 3.0))
        iqr_factor = float(qc_cfg.get('iqr_factor', 1.5))
        min_metrics = int(qc_cfg.get('min_metrics', 3))

        used_metrics = [m for m in self.metric_names if m in self.df.columns]
        if not used_metrics:
            meta = {
                'method': 'none',
                'reason': 'no_metric_columns',
                'n_subjects': int(len(self.df)),
                'n_outliers': 0,
                'outlier_indices': [],
                'inlier_mask': [True] * len(self.df),
                'metrics_used': [],
            }
            return self.df.copy(), meta

        x = self.df[used_metrics].replace([np.inf, -np.inf], np.nan).values
        valid_counts = np.isfinite(x).sum(axis=1)
        subject_mean = np.nanmean(x, axis=1)
        subject_std = np.nanstd(x, axis=1)
        enough_metric_mask = valid_counts >= max(1, min_metrics)
        inlier_mask = enough_metric_mask.copy()
        outlier_score = np.full(len(self.df), np.nan, dtype=float)

        if method == 'zscore':
            mu = np.nanmean(subject_mean[enough_metric_mask]) if enough_metric_mask.any() else 0.0
            sigma = np.nanstd(subject_mean[enough_metric_mask]) if enough_metric_mask.any() else 0.0
            if sigma <= 0 or not np.isfinite(sigma):
                z = np.zeros(len(subject_mean), dtype=float)
            else:
                z = np.abs((subject_mean - mu) / (sigma + 1e-10))
            outlier_score = z
            inlier_mask = (z < z_thr) & enough_metric_mask
        elif method == 'iqr':
            def _iqr_mask(vals, factor):
                v = vals[np.isfinite(vals)]
                if len(v) < 4:
                    return np.ones(len(vals), dtype=bool)
                q1, q3 = np.percentile(v, [25, 75])
                iqr = q3 - q1
                return (vals >= q1 - factor * iqr) & (vals <= q3 + factor * iqr)

            inlier_mask = _iqr_mask(subject_mean, iqr_factor) & \
                          _iqr_mask(subject_std, iqr_factor) & \
                          enough_metric_mask
        elif method == 'none':
            inlier_mask = enough_metric_mask
        else:
            raise ValueError(f"Unknown qc method '{method}'. Use: zscore, iqr, none")

        n_out = int((~inlier_mask).sum())
        meta = {
            'method': method,
            'zscore_threshold': z_thr if method == 'zscore' else None,
            'iqr_factor': iqr_factor if method == 'iqr' else None,
            'min_metrics': min_metrics,
            'n_subjects': int(len(self.df)),
            'n_outliers': n_out,
            'outlier_indices': np.where(~inlier_mask)[0].tolist(),
            'inlier_mask': inlier_mask.tolist(),
            'subject_metric_mean': subject_mean.tolist(),
            'subject_metric_std': subject_std.tolist(),
            'subject_valid_metric_count': valid_counts.tolist(),
            'subject_outlier_score': outlier_score.tolist(),
            'metrics_used': used_metrics,
        }

        logger.info(
            f"[qc] method={method}, outliers={n_out}/{len(self.df)} "
            f"({(n_out / max(len(self.df), 1)):.1%})"
        )

        df_qc = self.df.loc[inlier_mask].copy()
        df_qc.reset_index(drop=True, inplace=True)
        return df_qc, meta


# ================================================================
# Convenience Functions (for notebook / interactive use)
# ================================================================

def lifespan_trend(df: pd.DataFrame, metric_names: List[str],
                   cfg: dict = None) -> pd.DataFrame:
    """Compute age-metric correlations (convenience wrapper)."""
    return LifespanAnalyzer(df, metric_names, cfg).analyze()


def partial_corr_age(df: pd.DataFrame, metric_names: List[str],
                     covariates: List[str] = None,
                     cfg: dict = None) -> pd.DataFrame:
    """
    Partial correlations of metrics with age, controlling for covariates.

    Defaults to controlling for MDS_size.
    Compare r_zero vs r_partial to isolate structural age effects
    from those driven by network size/density changes.

    Example::

        results = partial_corr_age(df, metric_names)
        results = partial_corr_age(df, metric_names,
                                   covariates=['MDS_size_fraction', 'mean_degree'])
    """
    return PartialCorrelationAnalyzer(df, metric_names, covariates, cfg).analyze()


def assign_age_bins(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Assign age bins to subjects (convenience wrapper)."""
    return AgeBinAnalyzer(df, cfg).assign_bins()


def bin_summary_stats(df: pd.DataFrame, metric_names: List[str],
                      cfg: dict = None) -> pd.DataFrame:
    """Per-bin summary statistics (convenience wrapper)."""
    if cfg is None:
        cfg = {'age': {'bins': None, 'auto_bins': {'count': 5, 'method': 'equal'}}}
    return AgeBinAnalyzer(df, cfg).compute_bin_stats(df, metric_names)


def stat_tests(df: pd.DataFrame, metric_names: List[str],
               cfg: dict) -> Dict[str, dict]:
    """Per-metric statistical tests across age bins (convenience wrapper)."""
    return StatisticalTestAnalyzer(df, metric_names, cfg).run_tests()


def apply_subject_qc(df: pd.DataFrame, metric_names: List[str],
                      cfg: dict) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply subject-level QC (convenience wrapper)."""
    return QCAnalyzer(df, metric_names, cfg).run()
