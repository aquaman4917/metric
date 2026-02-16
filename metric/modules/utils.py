"""Utility functions: config loading, path management, parameter strings."""

import os
import yaml
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config and return as dict."""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded: {config_path}")
    return cfg


def override_config(cfg: dict, overrides: dict) -> dict:
    """
    Deep-merge overrides into cfg.
    Usage:
        cfg = override_config(cfg, {'network': {'density': 0.30}})
    """
    cfg = deepcopy(cfg)
    for key, val in overrides.items():
        if isinstance(val, dict) and key in cfg and isinstance(cfg[key], dict):
            cfg[key] = override_config(cfg[key], val)
        else:
            cfg[key] = val
    return cfg


def param_string(cfg: dict) -> str:
    """Config → filename-safe parameter string."""
    net = cfg['network']
    sub = cfg.get('subregion', {}).get('label', 'whole')
    return f"d{int(net['density']*100):02d}_f{int(net['frac']*100):02d}_n{net['net_size']}_{sub}"


def ensure_dir(path: str) -> str:
    """Create directory if not exists, return path."""
    os.makedirs(path, exist_ok=True)
    return path


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
