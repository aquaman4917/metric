"""
Brain Control Architecture Pipeline Modules
--------------------------------------------

Pipeline stages (the canonical entry points):

    Stage1Preprocessing  →  PreprocessedData
    Stage2Metrics        →  pd.DataFrame
    Stage3Analysis       →  AnalysisResults
    Stage4Visualization  →  figures + CSVs

Supporting modules (imported by stages internally):

    loader     – raw data I/O (.mat, subregions)
    network    – graph algorithms (thresholding, MDSet, community, PC)
    metrics    – metric definitions and registry
    analysis   – statistical analysis classes
    plotting   – visualization classes
    utils      – config, paths, shared helpers
"""

# ---- Stage classes (primary public API) ----
from .stage1_preprocessing import Stage1Preprocessing, PreprocessedData
from .stage2_metrics import Stage2Metrics
from .stage3_analysis import Stage3Analysis, AnalysisResults
from .stage4_visualization import Stage4Visualization
from .mds_conditional_zscore import MDSConditionalZScore, mds_conditional_zscore

# ---- Shared utilities ----
from .utils import (
    load_config,
    override_config,
    param_string,
    ensure_dir,
    setup_run_dirs,
    active_metrics,
    build_age_columns,
    print_metric_summary,
    setup_logging,
)

__all__ = [
    # Stages
    "Stage1Preprocessing",
    "Stage2Metrics",
    "Stage3Analysis",
    "Stage4Visualization",
    "MDSConditionalZScore",
    "mds_conditional_zscore",
    # Data containers
    "PreprocessedData",
    "AnalysisResults",
    # Utils
    "load_config",
    "override_config",
    "param_string",
    "ensure_dir",
    "setup_run_dirs",
    "active_metrics",
    "build_age_columns",
    "print_metric_summary",
    "setup_logging",
]
