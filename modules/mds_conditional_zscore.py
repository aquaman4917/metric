"""
MDS-size conditional z-score normalization.

This module computes within-MDS-size-bin z-scores:

    z_metric_i = (metric_i - mean(metric | MDS_bin)) / std(metric | MDS_bin)

Each subject is compared only to other subjects with similar MDS_size.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_METRICS = [
    'DC',
    'OCA',
    'OCA_norm',
    'OCA_P',
    'OCA_C',
    'DC_PC',
    'OCA_P_norm',
    'OCA_C_norm',
    'Prov_ratio',
    'Deg_conc',
    'newDC',
]


class MDSConditionalZScore:
    """
    Within-MDS-size-bin z-score normalization.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with MDS_size column and target metric columns.
    metrics : list[str], optional
        Metrics to normalize. Output columns are '{metric}_cz'.
    n_bins : int
        Number of quantile bins for MDS_size.
    min_bin_n : int
        Minimum subjects per bin to compute z-score.
    mds_col : str
        Column name for MDS_size.
    """

    def __init__(self, df: pd.DataFrame,
                 metrics: List[str] = None,
                 n_bins: int = 10,
                 min_bin_n: int = 20,
                 mds_col: str = 'MDS_size'):
        self.df = df.copy()
        self.metrics = [m for m in (metrics or DEFAULT_METRICS) if m in df.columns]
        self.n_bins = max(2, int(n_bins))
        self.min_bin_n = max(1, int(min_bin_n))
        self.mds_col = mds_col

        self.bin_edges_: Optional[np.ndarray] = None
        self.bin_stats_: Optional[pd.DataFrame] = None

    def _resolve_mds_col(self) -> Optional[str]:
        if self.mds_col in self.df.columns:
            return self.mds_col

        alt = 'MDS_size_fraction' if self.mds_col == 'MDS_size' else 'MDS_size'
        if alt in self.df.columns:
            logger.warning(
                "[cz] mds_col '%s' not found, falling back to '%s'",
                self.mds_col, alt,
            )
            self.mds_col = alt
            return alt

        logger.warning(
            "[cz] No MDS column found (requested '%s', fallback '%s'). "
            "All *_cz metrics will be NaN.",
            self.mds_col, alt,
        )
        return None

    def fit(self) -> 'MDSConditionalZScore':
        """
        Compute bin edges and per-bin statistics.
        Uses quantile binning so each bin has approximately equal N.
        """
        mds_col = self._resolve_mds_col()
        if mds_col is None:
            self.df['MDS_bin'] = -1
            self.bin_edges_ = np.array([], dtype=float)
            self.bin_stats_ = pd.DataFrame(columns=['MDS_bin', 'MDS_bin_lo', 'MDS_bin_hi', 'n'])
            return self

        mds = self.df[mds_col].to_numpy(dtype=float)
        valid = np.isfinite(mds)
        if valid.sum() == 0:
            logger.warning("[cz] No finite values in '%s'. All *_cz metrics will be NaN.", mds_col)
            self.df['MDS_bin'] = -1
            self.bin_edges_ = np.array([], dtype=float)
            self.bin_stats_ = pd.DataFrame(columns=['MDS_bin', 'MDS_bin_lo', 'MDS_bin_hi', 'n'])
            return self

        quantiles = np.linspace(0, 100, self.n_bins + 1)
        edges = np.percentile(mds[valid], quantiles)
        edges = np.unique(edges)
        if len(edges) < 2:
            v = float(mds[valid][0])
            edges = np.array([v - 1e-9, v + 1e-9], dtype=float)

        self.bin_edges_ = edges

        bin_idx = np.full(len(mds), -1, dtype=np.int32)
        bin_idx[valid] = np.digitize(mds[valid], edges[1:-1])
        self.df['MDS_bin'] = bin_idx

        rows = []
        for b in range(len(edges) - 1):
            mask = (bin_idx == b)
            n = int(mask.sum())
            lo = float(edges[b])
            hi = float(edges[b + 1])
            row = {'MDS_bin': b, 'MDS_bin_lo': lo, 'MDS_bin_hi': hi, 'n': n}

            for mname in self.metrics:
                vals = self.df.loc[mask, mname].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
                if len(vals) >= self.min_bin_n:
                    mu = float(np.mean(vals))
                    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
                    if np.isfinite(sd) and sd >= 1e-12:
                        row[f'{mname}_mean'] = mu
                        row[f'{mname}_std'] = sd
                    else:
                        row[f'{mname}_mean'] = np.nan
                        row[f'{mname}_std'] = np.nan
                        logger.warning(
                            "[cz] bin %d (%.4f-%.4f): near-zero std for %s, will produce NaN",
                            b, lo, hi, mname,
                        )
                else:
                    row[f'{mname}_mean'] = np.nan
                    row[f'{mname}_std'] = np.nan
                    logger.warning(
                        "[cz] bin %d (%.4f-%.4f): only %d valid for %s, will produce NaN",
                        b, lo, hi, len(vals), mname,
                    )

            rows.append(row)

        self.bin_stats_ = pd.DataFrame(rows)

        logger.info(
            "[cz] fit: %d bins over %s [%.4f, %.4f], metrics=%s",
            len(edges) - 1,
            self.mds_col,
            float(edges[0]),
            float(edges[-1]),
            self.metrics,
        )
        return self

    def transform(self) -> pd.DataFrame:
        """Add '{metric}_cz' columns to DataFrame. Requires fit() first."""
        if self.bin_stats_ is None:
            raise RuntimeError("Call fit() before transform()")

        df = self.df.copy()
        if self.bin_stats_.empty:
            for mname in self.metrics:
                df[f'{mname}_cz'] = np.nan
            return df

        stats_lookup = {}
        for _, row in self.bin_stats_.iterrows():
            b = int(row['MDS_bin'])
            stats_lookup[b] = {
                mname: (row.get(f'{mname}_mean', np.nan), row.get(f'{mname}_std', np.nan))
                for mname in self.metrics
            }

        for mname in self.metrics:
            cz_col = f'{mname}_cz'
            cz_vals = np.full(len(df), np.nan, dtype=float)

            for b, grp in df.groupby('MDS_bin'):
                b = int(b)
                if b < 0:
                    continue
                mu, sd = stats_lookup.get(b, {}).get(mname, (np.nan, np.nan))
                if not np.isfinite(mu) or not np.isfinite(sd) or sd < 1e-12:
                    continue
                idx = grp.index
                raw = df.loc[idx, mname].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
                cz_vals[idx] = (raw - mu) / sd

            df[cz_col] = cz_vals

            valid_cz = cz_vals[np.isfinite(cz_vals)]
            if len(valid_cz) > 0:
                logger.info(
                    "[cz] %s: n=%d, mean=%.3f, std=%.3f, nan=%d",
                    cz_col,
                    len(valid_cz),
                    float(np.mean(valid_cz)),
                    float(np.std(valid_cz)),
                    int(np.isnan(cz_vals).sum()),
                )
            else:
                logger.warning("[cz] %s: no valid z-scores produced", cz_col)

        return df

    def fit_transform(self) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit().transform()

    def bin_summary(self) -> pd.DataFrame:
        """Return bin statistics for diagnostics."""
        if self.bin_stats_ is None:
            raise RuntimeError("Call fit() first")
        return self.bin_stats_.copy()


def mds_conditional_zscore(df: pd.DataFrame,
                           metrics: List[str] = None,
                           n_bins: int = 10,
                           min_bin_n: int = 20,
                           mds_col: str = 'MDS_size') -> pd.DataFrame:
    """
    Add MDS_size-conditional z-score columns to DataFrame.

    Each '{metric}_cz' column is the z-score of that metric relative to
    subjects with similar MDS_size values.
    """
    return MDSConditionalZScore(
        df,
        metrics=metrics,
        n_bins=n_bins,
        min_bin_n=min_bin_n,
        mds_col=mds_col,
    ).fit_transform()
