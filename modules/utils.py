"""
Utility functions: config loading, path management, parameter strings, shared helpers.

This module centralises all cross-cutting concerns so that the four pipeline
stages (preprocessing → metrics → analysis → visualization) and main.py
never need to duplicate logic.
"""

import os
import yaml
import logging
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# ================================================================
# Config helpers
# ================================================================

def load_config(config_path: str) -> dict:
    """Load YAML config and return as dict."""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded: {config_path}")
    return cfg


def override_config(cfg: dict, overrides: dict) -> dict:
    """
    Deep-merge overrides into cfg.

    Usage::

        cfg = override_config(cfg, {'network': {'density': 0.30}})
    """
    cfg = deepcopy(cfg)
    for key, val in overrides.items():
        if isinstance(val, dict) and key in cfg and isinstance(cfg[key], dict):
            cfg[key] = override_config(cfg[key], val)
        else:
            cfg[key] = val
    return cfg


# ================================================================
# Parameter / path helpers
# ================================================================

def param_string(cfg: dict) -> str:
    """Config → filename-safe parameter string."""
    net = cfg['network']
    sub = cfg.get('subregion', {}).get('label', 'whole')
    return f"d{int(net['density']*100):02d}_f{int(net['frac']*100):02d}_n{net['net_size']}_{sub}"


def ensure_dir(path: str) -> str:
    """Create directory if not exists, return path."""
    os.makedirs(path, exist_ok=True)
    return path


def setup_run_dirs(cfg: dict) -> Dict[str, str]:
    """
    Create the full output directory tree for a single pipeline run.

    Returns:
        Dict with keys: run_dir, summary_dir, fig_dir, fig_detail_dir, param_str
    """
    pstr = param_string(cfg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{pstr}_{timestamp}"
    run_dir = ensure_dir(os.path.join(cfg['paths']['output_dir'], run_name))
    summary_dir = ensure_dir(os.path.join(run_dir, 'summary'))
    fig_dir = ensure_dir(os.path.join(summary_dir, 'figures'))
    fig_detail_dir = ensure_dir(os.path.join(fig_dir, 'detail'))
    return {
        'run_dir': run_dir,
        'summary_dir': summary_dir,
        'fig_dir': fig_dir,
        'fig_detail_dir': fig_detail_dir,
        'param_str': pstr,
    }


# ================================================================
# Metric helpers
# ================================================================

def active_metrics(cfg: dict) -> List[str]:
    """Return list of enabled metric names from config."""
    return [m for m, enabled in cfg['metrics'].items() if enabled]


# ================================================================
# DataFrame helpers
# ================================================================

def build_age_columns(df: pd.DataFrame, age: np.ndarray, cfg: dict) -> pd.DataFrame:
    """
    Insert age_years and age_months columns into *df* (in-place & returned).

    Args:
        df: DataFrame to augment
        age: Raw age array from data loader
        cfg: Config dict (uses ``cfg['age']['unit']``)

    Returns:
        Same DataFrame with age columns inserted at positions 0 and 1.
    """
    if cfg['age']['unit'] == 'month':
        df.insert(0, 'age_months', age)
        df.insert(1, 'age_years', age / 12.0)
    else:
        df.insert(0, 'age_years', age)
        df.insert(1, 'age_months', age * 12.0)
    return df


def print_metric_summary(df: pd.DataFrame, metric_names: List[str]):
    """Log descriptive statistics for each metric column."""
    n_subj = len(df)
    logger.info(f"\n{'='*50}")
    logger.info(f"  Summary: {n_subj} subjects, "
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
# Logging setup
# ================================================================

def setup_logging(level=logging.INFO, log_file: str = None):
    """
    Configure logging in consensus-style format.

    Args:
        level: Logging level (int or string)
        log_file: Optional file path for persistent logs
    """
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), logging.INFO)
    else:
        numeric_level = level

    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(numeric_level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
