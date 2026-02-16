"""
Stage 1: Data Loading & Network Preprocessing
==============================================
Pipeline:
  1. Load connectivity matrices and age data
  2. Optional subregion filtering
  3. Proportional thresholding
  4. Network analysis (degrees, MDSet)
  5. Cache preprocessed network features
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .loader import DataLoader, SubregionFilter
from .network import NetworkAnalyzer

logger = logging.getLogger(__name__)


# ================================================================
# Data Structures
# ================================================================

@dataclass
class PreprocessedData:
    """
    Container for preprocessed network data.

    Attributes:
        age: (N,) array of ages
        conn_raw: List of (M,M) raw connectivity matrices
        conn_binary: List of (M,M) binary adjacency matrices
        mdsets: List of MDSet node arrays
        top_nodes: List of high-degree node arrays
        n_subjects: Number of subjects
        n_nodes: Number of nodes (ROIs)
    """
    age: np.ndarray
    conn_raw: List[np.ndarray]
    conn_binary: List[np.ndarray]
    mdsets: List[np.ndarray]
    top_nodes: List[np.ndarray]
    n_subjects: int
    n_nodes: int

    def __repr__(self) -> str:
        return (f"PreprocessedData(n_subjects={self.n_subjects}, "
                f"n_nodes={self.n_nodes}, "
                f"age_range={self.age.min():.1f}~{self.age.max():.1f})")


# ================================================================
# Stage 1 Class
# ================================================================

class Stage1Preprocessing:
    """
    Stage 1: Data loading and network preprocessing.

    Handles all preprocessing steps before metric computation.
    """

    def __init__(self, cfg: dict):
        """
        Initialize preprocessing stage.

        Args:
            cfg: Full configuration dict
        """
        self.cfg = cfg
        self.data_loader = None
        self.subregion_filter = None
        self.network_analyzer = NetworkAnalyzer()

    @classmethod
    def run(cls, cfg: dict) -> PreprocessedData:
        """
        Execute Stage 1 preprocessing.

        Args:
            cfg: Configuration dict

        Returns:
            PreprocessedData container with all preprocessed features
        """
        stage = cls(cfg)
        return stage.execute()

    def execute(self) -> PreprocessedData:
        """
        Execute full preprocessing pipeline.

        Returns:
            PreprocessedData container
        """
        logger.info("="*60)
        logger.info("  Stage 1: Data Loading & Network Preprocessing")
        logger.info("="*60)

        # Step 1: Load data
        age, conn_raw, n_subjects = self._load_data()

        # Step 2: Apply subregion filter (if configured)
        conn_raw = self._apply_subregion_filter(conn_raw)
        n_nodes = conn_raw[0].shape[0]

        # Step 3: Preprocess all subjects
        conn_binary, mdsets, top_nodes = self._preprocess_all_subjects(conn_raw)

        # Create container
        data = PreprocessedData(
            age=age,
            conn_raw=conn_raw,
            conn_binary=conn_binary,
            mdsets=mdsets,
            top_nodes=top_nodes,
            n_subjects=n_subjects,
            n_nodes=n_nodes,
        )

        logger.info(f"[Stage1] Complete: {data}")
        return data

    def _load_data(self) -> Tuple[np.ndarray, List[np.ndarray], int]:
        """
        Load connectivity matrices and age data.

        Returns:
            (age, conn_list, n_subjects)
        """
        logger.info("[Step 1] Loading data...")

        self.data_loader = DataLoader(
            mat_path=self.cfg['paths']['input_mat'],
            age_unit=self.cfg['age']['unit'],
            search_dirs=self.cfg['paths'].get('input_search_dirs', [])
        )

        age, conn_list, n_subjects = self.data_loader.load()

        logger.info(f"[Step 1] Loaded {n_subjects} subjects, "
                   f"{conn_list[0].shape[0]} ROIs")

        return age, conn_list, n_subjects

    def _apply_subregion_filter(self, conn_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply subregion filtering if configured.

        Args:
            conn_list: List of connectivity matrices

        Returns:
            Filtered connectivity matrices
        """
        sub_nodes = self.cfg['subregion'].get('nodes')

        if sub_nodes is None:
            logger.info("[Step 2] No subregion filter")
            return conn_list

        logger.info(f"[Step 2] Applying subregion filter: {len(sub_nodes)} nodes")

        self.subregion_filter = SubregionFilter(sub_nodes)
        filtered = self.subregion_filter.filter(conn_list)

        # Update config with new network size
        self.cfg['network']['net_size'] = len(sub_nodes)

        return filtered

    def _preprocess_all_subjects(self, conn_list: List[np.ndarray]) -> Tuple[
        List[np.ndarray], List[np.ndarray], List[np.ndarray]
    ]:
        """
        Preprocess all subjects: threshold, compute MDSet, identify top nodes.

        Args:
            conn_list: List of raw connectivity matrices

        Returns:
            (conn_binary, mdsets, top_nodes)
        """
        logger.info("[Step 3] Preprocessing networks...")

        density = self.cfg['network']['density']
        frac = self.cfg['network']['frac']
        n_subjects = len(conn_list)

        conn_binary = []
        mdsets = []
        top_nodes = []

        for i, conn in enumerate(conn_list):
            # Proportional thresholding
            adj = self.network_analyzer.proportional_threshold(conn, density)

            # Create analyzer for this subject
            analyzer = NetworkAnalyzer(adj)

            # Compute MDSet
            mdset = analyzer.find_mdset(use_cache=True)

            # Identify high-degree nodes
            top = analyzer.top_degree_nodes(frac)

            conn_binary.append(adj)
            mdsets.append(mdset)
            top_nodes.append(top)

            if (i + 1) % 50 == 0:
                logger.info(f"[Step 3] Processed {i+1}/{n_subjects} subjects")

        logger.info(f"[Step 3] Complete: thresholded at density={density:.2f}, "
                   f"identified top {frac*100:.1f}% degree nodes")

        return conn_binary, mdsets, top_nodes


# ================================================================
# Standalone Function (Backward Compatibility)
# ================================================================

def run_preprocessing(cfg: dict) -> PreprocessedData:
    """
    Execute Stage 1 preprocessing (backward compatible).

    Args:
        cfg: Configuration dict

    Returns:
        PreprocessedData container
    """
    return Stage1Preprocessing.run(cfg)
