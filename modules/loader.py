"""
Data loading and subregion filtering.

Class-based data loading for brain connectivity matrices.
"""

import logging
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from typing import Tuple, Optional, List, Dict

logger = logging.getLogger(__name__)


# ================================================================
# Data Loader Class
# ================================================================

class DataLoader:
    """
    Loads connectivity matrices and age data from .mat files.

    Expected .mat format:
        Age:          (N,) or (N,1) array in months
        connectivity: (N,1) cell array of (246,246) matrices
    """

    def __init__(self, mat_path: str, age_unit: str = 'month',
                 search_dirs: Optional[List[str]] = None):
        """
        Initialize data loader.

        Args:
            mat_path: Path to .mat file
            age_unit: Unit of age in the file ('month' or 'year')
            search_dirs: Extra directories to search for .mat files
        """
        self.mat_path = mat_path
        self.age_unit = age_unit
        self.search_dirs = search_dirs or []
        self._data = None
        self._age = None
        self._conn_list = None
        self._n_subjects = None

    def load(self) -> Tuple[np.ndarray, List[np.ndarray], int]:
        """
        Load connectivity data from .mat file.

        Returns:
            age:       (N,) array of ages
            conn_list: List of (M,M) connectivity matrices
            n_subjects: Number of subjects
        """
        resolved_path = self._resolve_mat_path(self.mat_path)
        logger.info(f"Loading: {resolved_path}")

        data = loadmat(resolved_path, squeeze_me=False)
        self._data = data

        # Load age
        self._age = self._load_age(data)

        # Load connectivity matrices
        self._conn_list = self._load_connectivity(data)

        # Validate
        self._n_subjects = len(self._age)
        assert len(self._conn_list) == self._n_subjects, \
            f"Age ({self._n_subjects}) and connectivity ({len(self._conn_list)}) count mismatch"

        n = self._conn_list[0].shape[0]
        logger.info(f"Loaded: {self._n_subjects} subjects, {n}x{n} matrices, "
                   f"age range: {self._age.min():.0f}~{self._age.max():.0f} {self.age_unit}")

        return self._age, self._conn_list, self._n_subjects

    def _resolve_mat_path(self, mat_input: str) -> str:
        """
        Resolve MAT input from path or dataset/file name.

        Supported formats:
            - Absolute/relative path: data/brainnetome/HCP_merged.mat
            - File name: HCP_merged.mat
            - Stem name: HCP_merged

        Args:
            mat_input: User-provided path or name

        Returns:
            Resolved file path

        Raises:
            FileNotFoundError: If no matching file is found
            ValueError: If multiple files match the given name
        """
        if not mat_input:
            raise ValueError("Empty MAT input path/name")

        # 1) Direct path (absolute or relative)
        if os.path.isfile(mat_input):
            return mat_input

        base = os.path.basename(mat_input)
        stem, ext = os.path.splitext(base)
        if ext.lower() == '.mat':
            target_names = [base]
            target_stem = stem
        else:
            target_names = [f"{base}.mat"]
            target_stem = base

        # 2) Search in common roots
        roots = []
        for root in [os.getcwd(), 'data', os.path.join('data', 'brainnetome'), *self.search_dirs]:
            if root and root not in roots:
                roots.append(root)

        # 2a) Direct join lookup
        for root in roots:
            for name in target_names:
                cand = os.path.join(root, name)
                if os.path.isfile(cand):
                    return cand

        # 2b) Recursive exact stem lookup
        exact_matches = []
        for root in roots:
            if not os.path.isdir(root):
                continue
            for dirpath, _, filenames in os.walk(root):
                for fname in filenames:
                    if not fname.lower().endswith('.mat'):
                        continue
                    fstem, _ = os.path.splitext(fname)
                    if fname in target_names or fstem == target_stem:
                        exact_matches.append(os.path.join(dirpath, fname))

        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            raise ValueError(
                f"Ambiguous MAT input '{mat_input}'. Multiple matches found: {exact_matches}"
            )

        # 2c) Fuzzy stem contains lookup
        fuzzy_matches = []
        q = target_stem.lower()
        for root in roots:
            if not os.path.isdir(root):
                continue
            for dirpath, _, filenames in os.walk(root):
                for fname in filenames:
                    if not fname.lower().endswith('.mat'):
                        continue
                    fstem, _ = os.path.splitext(fname)
                    if q in fstem.lower():
                        fuzzy_matches.append(os.path.join(dirpath, fname))

        if len(fuzzy_matches) == 1:
            return fuzzy_matches[0]
        if len(fuzzy_matches) > 1:
            raise ValueError(
                f"Ambiguous MAT input '{mat_input}'. Candidates: {fuzzy_matches}"
            )

        available = []
        for root in roots:
            if not os.path.isdir(root):
                continue
            for dirpath, _, filenames in os.walk(root):
                for fname in filenames:
                    if fname.lower().endswith('.mat'):
                        available.append(os.path.join(dirpath, fname))

        raise FileNotFoundError(
            f"MAT file not found for input '{mat_input}'. "
            f"Available .mat files: {available if available else 'none'}"
        )

    def _load_age(self, data: dict) -> np.ndarray:
        """
        Extract age array from .mat data.

        Args:
            data: Loaded .mat dictionary

        Returns:
            (N,) age array

        Raises:
            KeyError: If 'Age' variable not found
        """
        for key in ['Age', 'age']:
            if key in data:
                age = np.asarray(data[key], dtype=np.float64).ravel()
                return age

        raise KeyError("'Age' variable not found in .mat file")

    def _load_connectivity(self, data: dict) -> List[np.ndarray]:
        """
        Extract connectivity matrices from .mat data.

        Args:
            data: Loaded .mat dictionary

        Returns:
            List of (M,M) connectivity matrices

        Raises:
            KeyError: If 'connectivity' variable not found
            ValueError: If format is unexpected
        """
        for key in ['connectivity', 'conn']:
            if key in data:
                raw = data[key]
                break
        else:
            raise KeyError("'connectivity' variable not found in .mat file")

        # Parse cell array → list of 2D arrays
        conn_list = []
        if raw.ndim == 2:  # (N,1) cell array
            for i in range(raw.shape[0]):
                mat = np.asarray(raw[i, 0], dtype=np.float64)
                conn_list.append(mat)
        else:
            raise ValueError(f"Unexpected connectivity shape: {raw.shape}")

        return conn_list

    @property
    def age(self) -> Optional[np.ndarray]:
        """Return loaded age data (or None if not loaded)."""
        return self._age

    @property
    def conn_list(self) -> Optional[List[np.ndarray]]:
        """Return loaded connectivity matrices (or None if not loaded)."""
        return self._conn_list

    @property
    def n_subjects(self) -> Optional[int]:
        """Return number of subjects (or None if not loaded)."""
        return self._n_subjects


# ================================================================
# Subregion Filter Class
# ================================================================

class SubregionFilter:
    """
    Filters connectivity matrices to extract subregions.

    Subregion is defined by a list of node indices.
    """

    def __init__(self, node_indices: Optional[List[int]] = None):
        """
        Initialize subregion filter.

        Args:
            node_indices: List of node indices to keep (None = no filtering)
        """
        self.node_indices = node_indices

    def filter(self, conn_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract subregion from connectivity matrices.

        Args:
            conn_list: List of (N,N) connectivity matrices

        Returns:
            List of (M,M) subregion connectivity matrices
        """
        if self.node_indices is None:
            return conn_list

        idx = np.array(self.node_indices)
        filtered = [c[np.ix_(idx, idx)] for c in conn_list]

        logger.info(f"Subregion filter: {conn_list[0].shape[0]} → {len(idx)} nodes")

        return filtered

    @property
    def n_nodes(self) -> Optional[int]:
        """Return number of nodes in subregion (or None if no filter)."""
        return len(self.node_indices) if self.node_indices is not None else None


# ================================================================
# Subregion Info Loader
# ================================================================

class SubregionInfoLoader:
    """
    Loads subregion information from Excel/CSV files.

    Useful for BNA parcellation or other anatomical atlases.
    """

    def __init__(self, info_path: str):
        """
        Initialize subregion info loader.

        Args:
            info_path: Path to Excel/CSV file with subregion info
        """
        self.info_path = info_path
        self._df = None

    def load(self) -> pd.DataFrame:
        """
        Load subregion info from file.

        Returns:
            DataFrame with subregion information
        """
        if self.info_path.endswith('.xlsx'):
            self._df = pd.read_excel(self.info_path)
        elif self.info_path.endswith('.csv'):
            self._df = pd.read_csv(self.info_path)
        else:
            raise ValueError(f"Unsupported file format: {self.info_path}")

        logger.info(f"Subregion info loaded: {len(self._df)} rows from {self.info_path}")
        return self._df

    def get_nodes_by_label(self, label: str,
                          column: str = 'lobe') -> List[int]:
        """
        Get node indices for a specific label.

        Args:
            label: Label to filter by (e.g., 'Frontal')
            column: Column name to filter on

        Returns:
            List of node indices
        """
        if self._df is None:
            self.load()

        mask = self._df[column] == label
        indices = self._df.loc[mask, 'index'].tolist()

        logger.info(f"Found {len(indices)} nodes for {column}='{label}'")
        return indices

    def get_all_labels(self, column: str = 'lobe') -> List[str]:
        """
        Get all unique labels in a column.

        Args:
            column: Column name

        Returns:
            List of unique labels
        """
        if self._df is None:
            self.load()

        return sorted(self._df[column].unique())

    def create_subregion_map(self, column: str = 'lobe') -> Dict[str, List[int]]:
        """
        Create mapping from labels to node indices.

        Args:
            column: Column name to group by

        Returns:
            Dict[label → node indices]
        """
        if self._df is None:
            self.load()

        labels = self.get_all_labels(column)
        return {
            label: self.get_nodes_by_label(label, column)
            for label in labels
        }


# ================================================================
# Standalone Functions (Backward Compatibility)
# ================================================================

def load_connectivity(cfg: dict) -> Tuple[np.ndarray, List[np.ndarray], int]:
    """
    Load connectivity .mat file (backward compatible).

    Args:
        cfg: Configuration dict with paths.input_mat and age.unit

    Returns:
        age:       (numSubj,) array
        conn_list: list of (N,N) ndarrays
        numSubj:   number of subjects
    """
    loader = DataLoader(
        mat_path=cfg['paths']['input_mat'],
        age_unit=cfg['age']['unit'],
        search_dirs=cfg['paths'].get('input_search_dirs', [])
    )
    return loader.load()


def filter_subregion(conn_list: List[np.ndarray],
                    node_indices: Optional[List[int]]) -> List[np.ndarray]:
    """
    Extract subregion from connectivity matrices (backward compatible).

    Args:
        conn_list: List of connectivity matrices
        node_indices: Node indices to keep (None = no filtering)

    Returns:
        List of filtered connectivity matrices
    """
    filter_obj = SubregionFilter(node_indices)
    return filter_obj.filter(conn_list)


def load_subregion_info(xlsx_path: str) -> pd.DataFrame:
    """
    Load BNA_subregions.xlsx for subregion node mapping (backward compatible).

    Args:
        xlsx_path: Path to Excel file

    Returns:
        DataFrame with columns like: index, label, lobe, hemisphere, etc.
    """
    loader = SubregionInfoLoader(xlsx_path)
    return loader.load()
