"""
Metric definitions and computation.

Class-based metric system for flexible experimentation.
Each metric is a self-contained class implementing BaseMetric interface.
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from . import network as net

logger = logging.getLogger(__name__)


# ================================================================
# Base Metric Class
# ================================================================

class BaseMetric(ABC):
    """
    Abstract base class for all metrics.

    All metrics must implement:
    - name: unique identifier
    - label: display label for plotting
    - compute(): computation logic
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique metric identifier (used in config and CSV columns)."""
        pass

    @property
    @abstractmethod
    def label(self) -> str:
        """Human-readable label for plotting."""
        pass

    @property
    def ref_line(self) -> Optional[float]:
        """Reference line value for plotting (e.g., DC = 1)."""
        return None

    @property
    def ref_text(self) -> str:
        """Reference line annotation text."""
        return ""

    @abstractmethod
    def compute(self, adj: np.ndarray, mdset: np.ndarray,
                top_nodes: np.ndarray, cfg: dict) -> float:
        """
        Compute metric value for a single subject.

        Args:
            adj: Binary adjacency matrix (N, N)
            mdset: Minimum dominating set node indices
            top_nodes: High-degree nodes (top FRAC)
            cfg: Full configuration dict

        Returns:
            Metric value (scalar)
        """
        pass

    def get_metadata(self) -> dict:
        """Return metadata for plotting/analysis."""
        return {
            'label': self.label,
            'ref_line': self.ref_line,
            'ref_text': self.ref_text,
        }


# ================================================================
# Individual Metric Implementations
# ================================================================

class MDSSizeMetric(BaseMetric):
    """MDS Size Fraction: |MDSet| / N"""

    @property
    def name(self) -> str:
        return 'MDS_size'

    @property
    def label(self) -> str:
        return 'MDS Size Fraction'

    def compute(self, adj: np.ndarray, mdset: np.ndarray,
                top_nodes: np.ndarray, cfg: dict) -> float:
        n = adj.shape[0]
        return len(mdset) / n if n > 0 else 0.0


class DegreeConcentrationMetric(BaseMetric):
    """Degree Concentration: overlap between MDSet and high-degree nodes"""

    @property
    def name(self) -> str:
        return 'Deg_conc'

    @property
    def label(self) -> str:
        return 'Degree Concentration'

    def compute(self, adj: np.ndarray, mdset: np.ndarray,
                top_nodes: np.ndarray, cfg: dict) -> float:
        overlap = np.intersect1d(mdset, top_nodes)
        return len(overlap) / len(top_nodes) if len(top_nodes) > 0 else 0.0


class DCMetric(BaseMetric):
    """
    Distribution of Control (DC).
    DC = |∪_{i ∈ MDSet_top} C(i)| / |∪_{i ∈ MDSet_bottom} C(i)|
    """

    @property
    def name(self) -> str:
        return 'DC'

    @property
    def label(self) -> str:
        return 'Distribution of Control (DC)'

    @property
    def ref_line(self) -> Optional[float]:
        return 1.0

    @property
    def ref_text(self) -> str:
        return 'DC = 1'

    def compute(self, adj: np.ndarray, mdset: np.ndarray,
                top_nodes: np.ndarray, cfg: dict) -> float:
        mdset_top = np.intersect1d(mdset, top_nodes)
        mdset_bottom = np.setdiff1d(mdset, top_nodes)

        area_top = net.union_control_area(adj, mdset_top)
        area_bottom = net.union_control_area(adj, mdset_bottom)

        if len(area_bottom) == 0:
            return np.inf
        return len(area_top) / len(area_bottom)


class OCAMetric(BaseMetric):
    """
    Overlap in Control Area (OCA).
    OCA = Σ|C(i)| / N
    """

    @property
    def name(self) -> str:
        return 'OCA'

    @property
    def label(self) -> str:
        return 'Overlap in Control Area (OCA)'

    @property
    def ref_line(self) -> Optional[float]:
        return 1.5

    @property
    def ref_text(self) -> str:
        return 'OCA = 1.5'

    def compute(self, adj: np.ndarray, mdset: np.ndarray,
                top_nodes: np.ndarray, cfg: dict) -> float:
        n = adj.shape[0]
        total = net.sum_control_area_sizes(adj, mdset)
        return total / n if n > 0 else 0.0


class NewDCMetric(BaseMetric):
    """
    newDC: FRAC value where DC crosses 1.0
    Finds the degree threshold where control becomes balanced.
    """

    @property
    def name(self) -> str:
        return 'newDC'

    @property
    def label(self) -> str:
        return 'newDC (FRAC where DC ≈ 1)'

    def compute(self, adj: np.ndarray, mdset: np.ndarray,
                top_nodes: np.ndarray, cfg: dict) -> float:
        # Search over FRAC range
        r = cfg['network']['frac_search_range']
        frac_range = np.arange(r[0], r[1] + r[2]/2, r[2])

        dc_vals = np.array([self._compute_dc_at_frac(adj, mdset, f)
                            for f in frac_range])

        # Filter valid
        valid = np.isfinite(dc_vals)
        if valid.sum() < 2:
            return np.nan

        fv = frac_range[valid]
        dv = dc_vals[valid]

        # Find crossing point (DC crosses 1)
        sign_change = np.diff(np.sign(dv - 1.0))
        cross_idx = np.where(sign_change != 0)[0]

        if len(cross_idx) > 0:
            # Linear interpolation at first crossing
            i = cross_idx[0]
            x1, x2 = fv[i], fv[i + 1]
            y1, y2 = dv[i], dv[i + 1]
            if y2 != y1:
                return x1 + (1.0 - y1) * (x2 - x1) / (y2 - y1)

        # No crossing → closest to DC=1
        closest = np.argmin(np.abs(dv - 1.0))
        return fv[closest]

    def _compute_dc_at_frac(self, adj: np.ndarray, mdset: np.ndarray,
                           frac: float) -> float:
        """Helper: compute DC at specific FRAC value."""
        top_nodes = net.top_degree_nodes(adj, frac)
        mdset_top = np.intersect1d(mdset, top_nodes)
        mdset_bottom = np.setdiff1d(mdset, top_nodes)

        area_top = net.union_control_area(adj, mdset_top)
        area_bottom = net.union_control_area(adj, mdset_bottom)

        if len(area_bottom) == 0:
            return np.inf
        return len(area_top) / len(area_bottom)


class DCPCMetric(BaseMetric):
    """
    DC_PC: Connector vs Provincial control areas.
    DC_PC = |∪ C(connector nodes)| / |∪ C(provincial nodes)|
    """

    @property
    def name(self) -> str:
        return 'DC_PC'

    @property
    def label(self) -> str:
        return r'$DC_{PC}$ (Connector / Provincial area)'

    @property
    def ref_line(self) -> Optional[float]:
        return 1.0

    @property
    def ref_text(self) -> str:
        return 'DC_PC = 1'

    def compute(self, adj: np.ndarray, mdset: np.ndarray,
                top_nodes: np.ndarray, cfg: dict) -> float:
        pc_results = self._compute_pc_classification(adj, mdset, cfg)
        return pc_results.get('DC_PC', np.nan)

    def _compute_pc_classification(self, adj: np.ndarray, mdset: np.ndarray,
                                   cfg: dict) -> dict:
        """Shared PC computation (cached to avoid redundant computation)."""
        n = adj.shape[0]
        pc_threshold = cfg['network']['pc_threshold']

        # Community detection
        communities = net.community_detection(adj)
        pc = net.participation_coefficient(adj, communities)

        # Classify MD-nodes
        prov_mask = pc[mdset] <= pc_threshold
        conn_mask = ~prov_mask

        prov_nodes = mdset[prov_mask]
        conn_nodes = mdset[conn_mask]

        # DC_PC = Connector union area / Provincial union area
        prov_area = net.union_control_area(adj, prov_nodes)
        conn_area = net.union_control_area(adj, conn_nodes)

        if len(prov_area) > 0:
            dc_pc = len(conn_area) / len(prov_area)
        else:
            dc_pc = np.inf

        # OCA_P and OCA_C
        oca_p = net.sum_control_area_sizes(adj, prov_nodes) / n if len(prov_nodes) > 0 else 0.0
        oca_c = net.sum_control_area_sizes(adj, conn_nodes) / n if len(conn_nodes) > 0 else 0.0

        # Provincial ratio
        prov_ratio = len(prov_nodes) / len(mdset) if len(mdset) > 0 else np.nan

        return {
            'DC_PC': dc_pc,
            'OCA_P': oca_p,
            'OCA_C': oca_c,
            'Prov_ratio': prov_ratio,
        }


class OCAPMetric(BaseMetric):
    """OCA for Provincial nodes only."""

    @property
    def name(self) -> str:
        return 'OCA_P'

    @property
    def label(self) -> str:
        return 'OCA_P (Provincial overlap)'

    def compute(self, adj: np.ndarray, mdset: np.ndarray,
                top_nodes: np.ndarray, cfg: dict) -> float:
        pc_results = DCPCMetric()._compute_pc_classification(adj, mdset, cfg)
        return pc_results.get('OCA_P', np.nan)


class OCACMetric(BaseMetric):
    """OCA for Connector nodes only."""

    @property
    def name(self) -> str:
        return 'OCA_C'

    @property
    def label(self) -> str:
        return 'OCA_C (Connector overlap)'

    def compute(self, adj: np.ndarray, mdset: np.ndarray,
                top_nodes: np.ndarray, cfg: dict) -> float:
        pc_results = DCPCMetric()._compute_pc_classification(adj, mdset, cfg)
        return pc_results.get('OCA_C', np.nan)


class ProvincialRatioMetric(BaseMetric):
    """Provincial Ratio: fraction of MD-nodes that are provincial."""

    @property
    def name(self) -> str:
        return 'Prov_ratio'

    @property
    def label(self) -> str:
        return 'Provincial Ratio'

    @property
    def ref_line(self) -> Optional[float]:
        return 0.6

    @property
    def ref_text(self) -> str:
        return '~60% (paper)'

    def compute(self, adj: np.ndarray, mdset: np.ndarray,
                top_nodes: np.ndarray, cfg: dict) -> float:
        pc_results = DCPCMetric()._compute_pc_classification(adj, mdset, cfg)
        return pc_results.get('Prov_ratio', np.nan)


# ================================================================
# Metric Registry
# ================================================================

class MetricRegistry:
    """
    Central registry for all available metrics.

    Manages metric instances and provides metadata for plotting/analysis.
    """

    def __init__(self):
        self._metrics: Dict[str, BaseMetric] = {}
        self._initialize_default_metrics()

    def _initialize_default_metrics(self):
        """Register all built-in metrics."""
        default_metrics = [
            MDSSizeMetric(),
            DegreeConcentrationMetric(),
            DCMetric(),
            OCAMetric(),
            NewDCMetric(),
            DCPCMetric(),
            OCAPMetric(),
            OCACMetric(),
            ProvincialRatioMetric(),
        ]

        for metric in default_metrics:
            self.register(metric)

    def register(self, metric: BaseMetric):
        """
        Register a metric in the registry.

        Args:
            metric: Metric instance implementing BaseMetric
        """
        self._metrics[metric.name] = metric
        logger.debug(f"Registered metric: {metric.name}")

    def get(self, name: str) -> Optional[BaseMetric]:
        """
        Get metric by name.

        Args:
            name: Metric identifier

        Returns:
            Metric instance or None if not found
        """
        return self._metrics.get(name)

    def get_all(self) -> Dict[str, BaseMetric]:
        """Return all registered metrics."""
        return self._metrics.copy()

    def get_metadata(self) -> dict:
        """
        Get metadata dict for all metrics (backward compatibility).

        Returns:
            Dict[metric_name → {label, ref_line, ref_text}]
        """
        return {
            name: metric.get_metadata()
            for name, metric in self._metrics.items()
        }


# ================================================================
# Metric Computer
# ================================================================

class MetricComputer:
    """
    Orchestrates computation of metrics for a single subject.

    Handles preprocessing (thresholding, MDSet) and computes
    only enabled metrics based on configuration.
    """

    def __init__(self, registry: MetricRegistry):
        """
        Initialize computer with metric registry.

        Args:
            registry: MetricRegistry instance with available metrics
        """
        self.registry = registry

    def compute_all(self, conn: np.ndarray, cfg: dict) -> Dict[str, float]:
        """
        Compute all enabled metrics for a single subject.

        Args:
            conn: Raw (N, N) connectivity matrix
            cfg: Full configuration dict

        Returns:
            Dict[metric_name → value]
        """
        net_cfg = cfg['network']
        metric_flags = cfg['metrics']

        # === Preprocessing ===
        # 1. Threshold → binary adjacency
        adj = net.proportional_threshold(conn, net_cfg['density'])

        # 2. Compute MDSet
        mdset = net.find_mdset(adj)

        # 3. Identify high-degree nodes
        top_nodes = net.top_degree_nodes(adj, net_cfg['frac'])

        # === Compute enabled metrics ===
        result = {}

        # Check for PC metrics to cache computation
        need_pc = any(metric_flags.get(m, False)
                     for m in ['DC_PC', 'OCA_P', 'OCA_C', 'Prov_ratio'])
        pc_cache = None

        if need_pc:
            # Compute PC classification once for all PC metrics
            pc_cache = self._compute_pc_cache(adj, mdset, cfg)

        for metric_name, enabled in metric_flags.items():
            if not enabled:
                continue

            metric = self.registry.get(metric_name)
            if metric is None:
                logger.warning(f"Metric '{metric_name}' enabled but not found in registry")
                continue

            try:
                # Use cached PC results if available
                if metric_name in ['DC_PC', 'OCA_P', 'OCA_C', 'Prov_ratio'] and pc_cache:
                    result[metric_name] = pc_cache[metric_name]
                else:
                    result[metric_name] = metric.compute(adj, mdset, top_nodes, cfg)
            except Exception as e:
                logger.error(f"Failed to compute {metric_name}: {e}")
                result[metric_name] = np.nan

        return result

    def _compute_pc_cache(self, adj: np.ndarray, mdset: np.ndarray,
                         cfg: dict) -> dict:
        """
        Compute PC-based classification once for all PC metrics.

        Returns:
            Dict with DC_PC, OCA_P, OCA_C, Prov_ratio
        """
        n = adj.shape[0]
        pc_threshold = cfg['network']['pc_threshold']

        # Community detection
        communities = net.community_detection(adj)
        pc = net.participation_coefficient(adj, communities)

        # Classify MD-nodes
        prov_mask = pc[mdset] <= pc_threshold
        conn_mask = ~prov_mask

        prov_nodes = mdset[prov_mask]
        conn_nodes = mdset[conn_mask]

        # Compute all PC metrics
        prov_area = net.union_control_area(adj, prov_nodes)
        conn_area = net.union_control_area(adj, conn_nodes)

        dc_pc = len(conn_area) / len(prov_area) if len(prov_area) > 0 else np.inf
        oca_p = net.sum_control_area_sizes(adj, prov_nodes) / n if len(prov_nodes) > 0 else 0.0
        oca_c = net.sum_control_area_sizes(adj, conn_nodes) / n if len(conn_nodes) > 0 else 0.0
        prov_ratio = len(prov_nodes) / len(mdset) if len(mdset) > 0 else np.nan

        return {
            'DC_PC': dc_pc,
            'OCA_P': oca_p,
            'OCA_C': oca_c,
            'Prov_ratio': prov_ratio,
        }


# ================================================================
# Public API (Backward Compatibility)
# ================================================================

# Global registry instance
_GLOBAL_REGISTRY = MetricRegistry()

# Expose metadata for backward compatibility
METRIC_REGISTRY = _GLOBAL_REGISTRY.get_metadata()


def compute_all_metrics(conn: np.ndarray, cfg: dict) -> Dict[str, float]:
    """
    Compute all enabled metrics for a single subject (backward compatible API).

    Args:
        conn: Raw (N, N) connectivity matrix
        cfg: Full configuration dict

    Returns:
        Dict[metric_name → value]
    """
    computer = MetricComputer(_GLOBAL_REGISTRY)
    return computer.compute_all(conn, cfg)


def get_metric_registry() -> MetricRegistry:
    """
    Get the global metric registry.

    Use this to add custom metrics:
        registry = get_metric_registry()
        registry.register(MyCustomMetric())
    """
    return _GLOBAL_REGISTRY
