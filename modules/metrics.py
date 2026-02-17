"""
Metric definitions and computation.

Decorator-based system: each metric is a plain function registered via @metric().
Add a new metric = write one function + one decorator call. That's it.

Example::

    @metric('My_metric', label='My Cool Metric', ref_line=0.5)
    def My_metric(adj, mdset, top_nodes, cfg):
        return some_value
"""
import logging
import numpy as np
from typing import Callable, Dict, Optional
from . import network as net

logger = logging.getLogger(__name__)


# ================================================================
# Registry (decorator-based)
# ================================================================

class MetricRegistry:
    """Stores metric functions + metadata.  Singleton via _REGISTRY."""

    def __init__(self):
        self._funcs: Dict[str, Callable] = {}
        self._meta: Dict[str, dict] = {}

    def register(self, name: str, fn: Callable, *,
                 label: str = '', ref_line: float = None, ref_text: str = ''):
        self._funcs[name] = fn
        self._meta[name] = {
            'label': label or name,
            'ref_line': ref_line,
            'ref_text': ref_text,
        }

    def get(self, name: str):
        fn = self._funcs.get(name)
        if fn is None:
            return None
        return _MetricHandle(name, fn, self._meta[name])

    def get_all(self):
        return {n: self.get(n) for n in self._funcs}

    def get_metadata(self) -> Dict[str, dict]:
        return dict(self._meta)

    def names(self):
        return list(self._funcs.keys())


class _MetricHandle:
    __slots__ = ('name', '_fn', '_meta')

    def __init__(self, name, fn, meta):
        self.name  = name
        self._fn   = fn
        self._meta = meta

    def compute(self, adj, mdset, top_nodes, cfg):
        return self._fn(adj, mdset, top_nodes, cfg)

    def get_metadata(self):
        return dict(self._meta)

    @property
    def label(self):    return self._meta['label']
    @property
    def ref_line(self): return self._meta['ref_line']
    @property
    def ref_text(self): return self._meta['ref_text']


_REGISTRY = MetricRegistry()


def metric(name: str, *, label: str = '', ref_line: float = None, ref_text: str = ''):
    """Decorator to register a metric function."""
    def _wrap(fn: Callable) -> Callable:
        _REGISTRY.register(name, fn, label=label, ref_line=ref_line, ref_text=ref_text)
        return fn
    return _wrap


# ================================================================
# Base metric definitions
# ================================================================

@metric('MDS_size', label='MDS Size Fraction')
def MDS_size(adj, mdset, top_nodes, cfg):
    """  |MDSet| / N  """
    n = adj.shape[0]
    return len(mdset) / n if n > 0 else 0.0


@metric('Deg_conc', label='Degree Concentration')
def Deg_conc(adj, mdset, top_nodes, cfg):
    """  MDSet ∩ high-degree overlap  """
    overlap = np.intersect1d(mdset, top_nodes)
    return len(overlap) / len(top_nodes) if len(top_nodes) > 0 else 0.0


@metric('DC', label='Distribution of Control (DC)', ref_line=1.0, ref_text='DC = 1')
def DC(adj, mdset, top_nodes, cfg):
    """  |∪ C(top)| / |∪ C(bottom)|  """
    mdset_top    = np.intersect1d(mdset, top_nodes)
    mdset_bottom = np.setdiff1d(mdset, top_nodes)
    area_top     = net.union_control_area(adj, mdset_top)
    area_bottom  = net.union_control_area(adj, mdset_bottom)
    if len(area_bottom) == 0:
        return np.inf
    return len(area_top) / len(area_bottom)


@metric('OCA', label='Overlap in Control Area (OCA)', ref_line=1.5, ref_text='OCA = 1.5')
def OCA(adj, mdset, top_nodes, cfg):
    """  Σ|C(i)| / N  """
    n     = adj.shape[0]
    total = net.sum_control_area_sizes(adj, mdset)
    return total / n if n > 0 else 0.0


@metric('OCA_norm', label='OCA / MDS_size (mean control area per MD-node)')
def OCA_norm(adj, mdset, top_nodes, cfg):
    """
    OCA normalised by MDS_size.

    Derivation:
        OCA      = Σ|C(i)| / N
        MDS_size = M / N
        OCA_norm = OCA / MDS_size = Σ|C(i)| / M

    Interpretation: average number of nodes dominated by a single MD-node.
    This is the per-MD-node control efficiency, independent of how many
    MD-nodes exist.  Two subjects with different MDS_size values are now
    directly comparable on this metric.

    Note: OCA_norm ≥ 1 always (every MD-node dominates at least itself).
    """
    m = len(mdset)
    if m == 0:
        return np.nan
    total = net.sum_control_area_sizes(adj, mdset)
    return total / m   # = Σ|C(i)| / M


###############################################################################
# New Metric: Normalized Overlap Redundancy (NOR)
###############################################################################

@metric('NOR', label='Normalized Overlap Redundancy')
def NOR(adj, mdset, top_nodes, cfg):
    """
    Normalized Overlap Redundancy (NOR) measures overlap redundancy independent of MDS size.
    Defined as (OCA - 1) / (M - 1), where
    OCA = Σ|C(i)| / N and M = |MDSet|.
    Returns NaN if M <= 1.
    """
    n = adj.shape[0]
    m = len(mdset)
    if m <= 1:
        return np.nan
    total = net.sum_control_area_sizes(adj, mdset)
    # Compute OCA (mean total control area per node)
    oca = total / n if n > 0 else np.nan
    return (oca - 1.0) / (m - 1.0)


@metric('newDC', label='newDC (FRAC where DC ≈ 1)')
def newDC(adj, mdset, top_nodes, cfg):
    """FRAC value where DC crosses 1.0 (linear interpolation)."""
    r          = cfg['network']['frac_search_range']
    frac_range = np.arange(r[0], r[1] + r[2] / 2, r[2])
    dc_vals    = np.array([_dc_at_frac(adj, mdset, f) for f in frac_range])

    valid = np.isfinite(dc_vals)
    if valid.sum() < 2:
        return np.nan

    fv, dv      = frac_range[valid], dc_vals[valid]
    sign_change = np.diff(np.sign(dv - 1.0))
    cross_idx   = np.where(sign_change != 0)[0]

    if len(cross_idx) > 0:
        i       = cross_idx[0]
        x1, x2  = fv[i], fv[i + 1]
        y1, y2  = dv[i], dv[i + 1]
        if y2 != y1:
            return x1 + (1.0 - y1) * (x2 - x1) / (y2 - y1)

    return fv[np.argmin(np.abs(dv - 1.0))]


def _dc_at_frac(adj, mdset, frac):
    top        = net.top_degree_nodes(adj, frac)
    mdset_top  = np.intersect1d(mdset, top)
    mdset_bot  = np.setdiff1d(mdset, top)
    area_top   = net.union_control_area(adj, mdset_top)
    area_bot   = net.union_control_area(adj, mdset_bot)
    if len(area_bot) == 0:
        return np.inf
    return len(area_top) / len(area_bot)


# ================================================================
# PC-based metrics
# ================================================================

def _pc_classify(adj, mdset, cfg):
    """Provincial / Connector classification of MD-nodes. Returns dict."""
    n            = adj.shape[0]
    pc_threshold = cfg['network']['pc_threshold']
    communities  = net.community_detection(adj)
    pc           = net.participation_coefficient(adj, communities)

    prov_mask  = pc[mdset] <= pc_threshold
    prov_nodes = mdset[prov_mask]
    conn_nodes = mdset[~prov_mask]

    prov_area = net.union_control_area(adj, prov_nodes)
    conn_area = net.union_control_area(adj, conn_nodes)
    m         = len(mdset)

    return {
        'DC_PC':       len(conn_area) / len(prov_area) if len(prov_area) > 0 else np.inf,
        'OCA_P':       net.sum_control_area_sizes(adj, prov_nodes) / n if len(prov_nodes) > 0 else 0.0,
        'OCA_C':       net.sum_control_area_sizes(adj, conn_nodes) / n if len(conn_nodes) > 0 else 0.0,
        'Prov_ratio':  len(prov_nodes) / m if m > 0 else np.nan,
        # MDS_size-normalised variants: Σ|C(i)| / M_subgroup
        # i.e. mean control area per Provincial / Connector MD-node
        'OCA_P_norm':  net.sum_control_area_sizes(adj, prov_nodes) / len(prov_nodes) if len(prov_nodes) > 0 else np.nan,
        'OCA_C_norm':  net.sum_control_area_sizes(adj, conn_nodes) / len(conn_nodes) if len(conn_nodes) > 0 else np.nan,
    }


@metric('DC_PC', label=r'$DC_{PC}$ (Connector / Provincial area)',
        ref_line=1.0, ref_text='DC_PC = 1')
def DC_PC(adj, mdset, top_nodes, cfg):
    return _pc_classify(adj, mdset, cfg)['DC_PC']


@metric('OCA_P', label='OCA_P (Provincial overlap)')
def OCA_P(adj, mdset, top_nodes, cfg):
    return _pc_classify(adj, mdset, cfg)['OCA_P']


@metric('OCA_C', label='OCA_C (Connector overlap)')
def OCA_C(adj, mdset, top_nodes, cfg):
    return _pc_classify(adj, mdset, cfg)['OCA_C']


@metric('Prov_ratio', label='Provincial Ratio', ref_line=0.6, ref_text='~60% (paper)')
def Prov_ratio(adj, mdset, top_nodes, cfg):
    return _pc_classify(adj, mdset, cfg)['Prov_ratio']


@metric('OCA_P_norm', label='OCA_P_norm (mean control area per Provincial MD-node)')
def OCA_P_norm(adj, mdset, top_nodes, cfg):
    return _pc_classify(adj, mdset, cfg)['OCA_P_norm']


@metric('OCA_C_norm', label='OCA_C_norm (mean control area per Connector MD-node)')
def OCA_C_norm(adj, mdset, top_nodes, cfg):
    return _pc_classify(adj, mdset, cfg)['OCA_C_norm']


# ================================================================
# Null-model infrastructure (degree-preserving, Maslov–Sneppen)
# ================================================================

def _edges_upper(adj: np.ndarray) -> np.ndarray:
    iu   = np.triu_indices_from(adj, k=1)
    mask = adj[iu] > 0
    return np.vstack((iu[0][mask], iu[1][mask])).T


def _maslov_sneppen_swaps(adj: np.ndarray, swaps_per_edge: int = 10,
                           rng: np.random.Generator = None) -> np.ndarray:
    """
    Degree-preserving null via Maslov–Sneppen edge swaps.
    Input/output: full symmetric 0/1 adj, no self-loops.
    """
    if rng is None:
        rng = np.random.default_rng()

    A0    = np.array(adj, dtype=np.int8, copy=True)
    np.fill_diagonal(A0, 0)
    A     = np.triu(A0, 1).copy()
    edges = _edges_upper(A)
    m     = edges.shape[0]
    if m < 2:
        return A + A.T

    for _ in range(int(swaps_per_edge * m)):
        i1 = int(rng.integers(0, m))
        i2 = int(rng.integers(0, m))
        if i1 == i2:
            continue

        a, b = edges[i1]
        c, d = edges[i2]
        if len({a, b, c, d}) < 4:
            continue

        u1, v1 = (a, d) if a < d else (d, a)
        u2, v2 = (c, b) if c < b else (b, c)
        if u1 == v1 or u2 == v2:
            continue
        if A[u1, v1] == 1 or A[u2, v2] == 1:
            continue

        uo1, vo1 = (a, b) if a < b else (b, a)
        uo2, vo2 = (c, d) if c < d else (d, c)
        A[uo1, vo1] = 0;  A[uo2, vo2] = 0
        A[u1,  v1]  = 1;  A[u2,  v2]  = 1
        edges[i1] = (u1, v1)
        edges[i2] = (u2, v2)

    return A + A.T


def _null_zscore(adj: np.ndarray, cfg: dict, base_fn: Callable,
                 n_null: int = 100, swaps_per_edge: int = 10,
                 seed: Optional[int] = None) -> float:
    """
    z-score of base_fn against degree-preserving null networks.

    Each null gets its own MDSet + top_nodes recomputed, so the z-score
    reflects structural deviation beyond what degree sequence predicts.

    base_fn: (adj, mdset, top_nodes, cfg) -> float
    """
    rng     = np.random.default_rng(seed)
    md_obs  = net.find_mdset(adj)
    top_obs = net.top_degree_nodes(adj, cfg['network']['frac'])
    x       = base_fn(adj, md_obs, top_obs, cfg)

    null_vals = []
    for _ in range(int(n_null)):
        adj_null = _maslov_sneppen_swaps(adj, swaps_per_edge=swaps_per_edge, rng=rng)
        md_n     = net.find_mdset(adj_null)
        top_n    = net.top_degree_nodes(adj_null, cfg['network']['frac'])
        v        = base_fn(adj_null, md_n, top_n, cfg)
        if np.isfinite(v):
            null_vals.append(float(v))

    if (not np.isfinite(x)) or len(null_vals) < 10:
        return np.nan

    mu = float(np.mean(null_vals))
    sd = float(np.std(null_vals, ddof=1))
    if sd <= 1e-12:
        return np.nan
    return (float(x) - mu) / sd


@metric('DC_z', label='DC z-score (degree-preserving null)')
def DC_z(adj, mdset, top_nodes, cfg):
    n_null = cfg['network'].get('null_n', 100)
    swaps  = cfg['network'].get('null_swaps_per_edge', 10)
    seed   = cfg['network'].get('null_seed', None)
    return _null_zscore(adj, cfg, DC, n_null=n_null, swaps_per_edge=swaps, seed=seed)


@metric('OCA_z', label='OCA z-score (degree-preserving null)')
def OCA_z(adj, mdset, top_nodes, cfg):
    n_null = cfg['network'].get('null_n', 100)
    swaps  = cfg['network'].get('null_swaps_per_edge', 10)
    seed   = cfg['network'].get('null_seed', None)
    return _null_zscore(adj, cfg, OCA, n_null=n_null, swaps_per_edge=swaps, seed=seed)


@metric('OCA_norm_z', label='OCA_norm z-score (degree-preserving null)')
def OCA_norm_z(adj, mdset, top_nodes, cfg):
    """
    z-score of OCA_norm against degree-preserving nulls.

    OCA_norm = mean control area per MD-node (MDS_size-independent).
    Null networks have the same degree sequence but different topology,
    and naturally different MDS_sizes.  By normalising both observed and
    null values by their own MDS_size before z-scoring, this isolates
    whether the per-node control efficiency is structurally special
    beyond what degree alone predicts — without MDS_size confound.
    """
    n_null = cfg['network'].get('null_n', 100)
    swaps  = cfg['network'].get('null_swaps_per_edge', 10)
    seed   = cfg['network'].get('null_seed', None)
    return _null_zscore(adj, cfg, OCA_norm, n_null=n_null, swaps_per_edge=swaps, seed=seed)


# ================================================================
# MetricComputer — orchestrates per-subject computation
# ================================================================

PC_METRICS = frozenset(['DC_PC', 'OCA_P', 'OCA_C', 'Prov_ratio',
                        'OCA_P_norm', 'OCA_C_norm'])


class MetricComputer:
    """Compute all enabled metrics for one subject."""

    def __init__(self, registry: MetricRegistry = None):
        self.registry = registry or _REGISTRY

    def compute_all(self, conn: np.ndarray, cfg: dict) -> Dict[str, float]:
        net_cfg   = cfg['network']
        adj       = net.proportional_threshold(conn, net_cfg['density'])
        mdset     = net.find_mdset(adj)
        top_nodes = net.top_degree_nodes(adj, net_cfg['frac'])
        return self.compute_all_from_preprocessed(adj, mdset, top_nodes, cfg)

    def compute_all_from_preprocessed(self, adj: np.ndarray, mdset: np.ndarray,
                                       top_nodes: np.ndarray, cfg: dict) -> Dict[str, float]:
        flags    = cfg['metrics']
        pc_cache = None
        if any(flags.get(m, False) for m in PC_METRICS):
            pc_cache = _pc_classify(adj, mdset, cfg)

        result = {}
        for mname, enabled in flags.items():
            if not enabled:
                continue
            # Derived cohort-level metrics are computed after Stage2 aggregation.
            if mname.endswith('_cz'):
                continue
            handle = self.registry.get(mname)
            if handle is None:
                logger.warning(f"Metric '{mname}' enabled but not registered")
                continue
            try:
                if mname in PC_METRICS and pc_cache is not None:
                    result[mname] = pc_cache[mname]
                else:
                    result[mname] = handle.compute(adj, mdset, top_nodes, cfg)
            except Exception as e:
                logger.error(f"Failed to compute {mname}: {e}")
                result[mname] = np.nan

        return result


# ================================================================
# Public API
# ================================================================

METRIC_REGISTRY = _REGISTRY.get_metadata()


def compute_all_metrics(conn: np.ndarray, cfg: dict) -> Dict[str, float]:
    """Compute all enabled metrics for one subject (convenience)."""
    return MetricComputer(_REGISTRY).compute_all(conn, cfg)


def get_metric_registry() -> MetricRegistry:
    """Get the global registry (to add custom metrics at runtime)."""
    return _REGISTRY
