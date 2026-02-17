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
        Preprocess all subjects: threshold → MDSet → top nodes.

        Returns:
            (conn_binary, mdsets, top_nodes)
        """
        logger.info("[Step 3] Preprocessing networks...")

        n_subjects = len(conn_list)
        conn_binary = []
        mdsets = []
        top_nodes = []

        for i, conn in enumerate(conn_list):
            adj = self._threshold(conn)
            mdset = self._compute_mdset(adj)
            top = self._compute_top_nodes(adj)

            conn_binary.append(adj)
            mdsets.append(mdset)
            top_nodes.append(top)

            if (i + 1) % 50 == 0:
                logger.info(f"[Step 3] Processed {i+1}/{n_subjects} subjects")

        density = self.cfg['network']['density']
        frac = self.cfg['network']['frac']
        logger.info(f"[Step 3] Complete: density={density:.2f}, "
                   f"top {frac*100:.1f}% degree nodes")

        return conn_binary, mdsets, top_nodes

    # ------------------------------------------------------------------
    # Internal steps — override / swap these to change preprocessing
    # ------------------------------------------------------------------

    def _threshold(self, conn: np.ndarray) -> np.ndarray:
        """Weighted connectivity → binary adjacency.

        Dispatches on ``cfg['preprocessing']['threshold_method']``:

            * ``"proportional"`` — top *density* % edges (default)
            * ``"absolute"``     — edges with |w| >= fixed value
            * ``"mst"``          — MST backbone + proportional fill
        """
        pre_cfg = self.cfg.get('preprocessing', {})
        method = pre_cfg.get('threshold_method', 'proportional')

        dispatch = {
            'proportional': self._threshold_proportional,
            'absolute': self._threshold_absolute,
            'mst': self._threshold_mst,
        }

        fn = dispatch.get(method)
        if fn is None:
            raise ValueError(
                f"Unknown threshold_method '{method}'. "
                f"Choose from: {list(dispatch.keys())}"
            )
        return fn(conn)

    # --- threshold implementations ---

    def _threshold_proportional(self, conn: np.ndarray) -> np.ndarray:
        """Keep top *density* fraction of strongest edges."""
        density = self.cfg['network']['density']
        return self.network_analyzer.proportional_threshold(conn, density)

    def _threshold_absolute(self, conn: np.ndarray) -> np.ndarray:
        """Keep edges whose |weight| >= a fixed threshold."""
        pre_cfg = self.cfg.get('preprocessing', {})
        thr = pre_cfg.get('absolute_threshold')
        if thr is None:
            raise ValueError(
                "absolute_threshold must be set when threshold_method='absolute'"
            )
        n = conn.shape[0]
        w = (conn + conn.T) / 2.0
        np.fill_diagonal(w, 0)
        binary = (np.abs(w) >= thr).astype(np.int32)
        return np.maximum(binary, binary.T)

    def _threshold_mst(self, conn: np.ndarray) -> np.ndarray:
        """MST backbone, then fill remaining budget with strongest edges."""
        from scipy.sparse.csgraph import minimum_spanning_tree

        density = self.cfg['network']['density']
        n = conn.shape[0]

        w = (conn + conn.T) / 2.0
        np.fill_diagonal(w, 0)
        abs_w = np.abs(w)

        # MST (scipy minimises → negate to get max-weight tree)
        mst = minimum_spanning_tree(-abs_w)
        mst_bin = (mst.toarray() != 0).astype(np.int32)
        mst_bin = np.maximum(mst_bin, mst_bin.T)

        # Fill to target density
        triu_idx = np.triu_indices(n, k=1)
        n_target = max(1, int(np.round(density * len(triu_idx[0]))))
        n_mst = mst_bin[triu_idx].sum()
        n_fill = max(0, n_target - n_mst)

        if n_fill > 0:
            non_mst_mask = mst_bin[triu_idx] == 0
            non_mst_weights = abs_w[triu_idx] * non_mst_mask
            fill_idx = np.argsort(non_mst_weights)[::-1][:n_fill]
            for idx in fill_idx:
                r, c = triu_idx[0][idx], triu_idx[1][idx]
                mst_bin[r, c] = 1
                mst_bin[c, r] = 1

        return mst_bin

    # --- other preprocessing steps ---

    def _compute_mdset(self, adj: np.ndarray) -> np.ndarray:
        """Binary adjacency → Minimum Dominating Set.

        Reads ``cfg['preprocessing']['mdset_method']``:
            * ``'greedy'`` — fast heuristic (~5 ms / subject).  Default.
            * ``'ilp'``    — exact ILP solver (minutes / subject).
        """
        pre_cfg = self.cfg.get('preprocessing', {})
        method = pre_cfg.get('mdset_method', 'greedy')
        analyzer = NetworkAnalyzer(adj)
        return analyzer.find_mdset(use_cache=True, method=method)

    def _compute_top_nodes(self, adj: np.ndarray) -> np.ndarray:
        """Binary adjacency → high-degree node indices (top *frac* %)."""
        frac = self.cfg['network']['frac']
        analyzer = NetworkAnalyzer(adj)
        return analyzer.top_degree_nodes(frac)


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
