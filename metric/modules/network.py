"""
Network operations — replaces BCT (Brain Connectivity Toolbox).

Class-based network analysis for brain connectivity.
All functions operate on numpy arrays directly.
MDSet uses PuLP for binary integer programming (branch-and-bound).
Community detection uses greedy modularity maximization.
"""

import logging
import numpy as np
from scipy import sparse
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


# ================================================================
# Network Analyzer Class
# ================================================================

class NetworkAnalyzer:
    """
    Main class for brain network analysis operations.

    Provides methods for:
    - Thresholding and binarization
    - Degree analysis
    - Minimum Dominating Set (MDSet) computation
    - Control area analysis
    - Community detection
    - Participation coefficient
    """

    def __init__(self, adj: Optional[np.ndarray] = None):
        """
        Initialize network analyzer.

        Args:
            adj: Optional (N, N) binary adjacency matrix
        """
        self.adj = adj
        self._mdset_cache = None
        self._communities_cache = None

    # ================================================================
    # 1. Thresholding
    # ================================================================

    @staticmethod
    def proportional_threshold(conn: np.ndarray, density: float) -> np.ndarray:
        """
        Proportional thresholding → binary undirected matrix.

        Keeps top `density` fraction of strongest connections (by absolute weight).
        Diagonal is zeroed. Output is symmetric binary.

        Args:
            conn: (N, N) weighted connectivity matrix
            density: fraction of edges to keep (e.g., 0.20 = top 20%)

        Returns:
            (N, N) binary adjacency matrix
        """
        n = conn.shape[0]

        # Symmetrize + zero diagonal
        w = (conn + conn.T) / 2.0
        np.fill_diagonal(w, 0)

        # Upper triangle weights
        triu_idx = np.triu_indices(n, k=1)
        weights = np.abs(w[triu_idx])

        # Threshold
        n_edges_total = len(weights)
        n_keep = max(1, int(np.round(density * n_edges_total)))
        threshold = np.sort(weights)[::-1][min(n_keep - 1, n_edges_total - 1)]

        # Binary matrix
        binary = np.zeros((n, n), dtype=np.int32)
        mask = np.abs(w) >= threshold
        np.fill_diagonal(mask, False)
        binary[mask] = 1

        # Ensure symmetric
        binary = np.maximum(binary, binary.T)
        return binary

    # ================================================================
    # 2. Degree Analysis
    # ================================================================

    def degrees(self, adj: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute node degrees from binary adjacency matrix.

        Args:
            adj: Optional adjacency matrix (uses self.adj if None)

        Returns:
            (N,) array of node degrees
        """
        if adj is None:
            adj = self.adj
        if adj is None:
            raise ValueError("No adjacency matrix provided")

        return adj.sum(axis=1).astype(np.int32)

    def top_degree_nodes(self, frac: float,
                        adj: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Indices of top `frac` fraction high-degree nodes.

        Uses threshold: nodes with degree >= the k-th highest degree,
        where k = floor(frac * N).

        Args:
            frac: Fraction of top nodes to select
            adj: Optional adjacency matrix (uses self.adj if None)

        Returns:
            Array of node indices
        """
        if adj is None:
            adj = self.adj
        if adj is None:
            raise ValueError("No adjacency matrix provided")

        deg = self.degrees(adj)
        n = len(deg)
        k = max(1, min(n, int(np.floor(frac * n))))

        # k-th highest degree value
        sorted_deg = np.sort(deg)[::-1]
        thr = sorted_deg[min(k - 1, n - 1)]

        return np.where(deg >= thr)[0]

    # ================================================================
    # 3. Minimum Dominating Set (ILP)
    # ================================================================

    def find_mdset(self, adj: Optional[np.ndarray] = None,
                  use_cache: bool = True) -> np.ndarray:
        """
        Find Minimum Dominating Set using binary integer programming.

        Formulation (Lee et al. 2019):
            min  Σ x_v
            s.t. x_v + Σ_{w ∈ Γ(v)} x_w >= 1  ∀v
                 x_v ∈ {0, 1}

        Args:
            adj: Optional adjacency matrix (uses self.adj if None)
            use_cache: If True, cache result for repeated calls

        Returns:
            1D array of MDSet node indices (0-indexed)
        """
        if adj is None:
            adj = self.adj
        if adj is None:
            raise ValueError("No adjacency matrix provided")

        # Return cached result if available
        if use_cache and self._mdset_cache is not None:
            return self._mdset_cache

        import pulp

        n = adj.shape[0]

        # Create problem
        prob = pulp.LpProblem("MDSet", pulp.LpMinimize)

        # Binary variables
        x = [pulp.LpVariable(f"x_{i}", cat='Binary') for i in range(n)]

        # Objective: minimize total selected nodes
        prob += pulp.lpSum(x)

        # Constraints: each node must be dominated
        for v in range(n):
            neighbors = np.where(adj[v] > 0)[0]
            prob += x[v] + pulp.lpSum(x[w] for w in neighbors) >= 1

        # Solve (suppress output)
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if prob.status != 1:
            logger.warning(f"MDSet solver status: {pulp.LpStatus[prob.status]}")

        mdset = np.array([i for i in range(n) if x[i].varValue > 0.5],
                        dtype=np.int32)

        # Cache if requested
        if use_cache:
            self._mdset_cache = mdset

        return mdset

    # ================================================================
    # 4. Control Area
    # ================================================================

    def control_area(self, node: int,
                    adj: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Nodes dominated by a single MD-node (self + direct neighbors).

        Args:
            node: Node index
            adj: Optional adjacency matrix (uses self.adj if None)

        Returns:
            Array of dominated node indices
        """
        if adj is None:
            adj = self.adj
        if adj is None:
            raise ValueError("No adjacency matrix provided")

        neighbors = np.where(adj[node] > 0)[0]
        return np.union1d([node], neighbors)

    def union_control_area(self, nodes: np.ndarray,
                          adj: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Union of control areas of multiple nodes.

        Args:
            nodes: Array of node indices
            adj: Optional adjacency matrix (uses self.adj if None)

        Returns:
            Array of unique dominated node indices
        """
        if adj is None:
            adj = self.adj
        if adj is None:
            raise ValueError("No adjacency matrix provided")

        if len(nodes) == 0:
            return np.array([], dtype=np.int32)

        dominated = set()
        for nd in nodes:
            dominated.add(nd)
            dominated.update(np.where(adj[nd] > 0)[0])

        return np.array(sorted(dominated), dtype=np.int32)

    def sum_control_area_sizes(self, nodes: np.ndarray,
                              adj: Optional[np.ndarray] = None) -> int:
        """
        Sum of individual control area sizes (allows overlap counting).

        Args:
            nodes: Array of node indices
            adj: Optional adjacency matrix (uses self.adj if None)

        Returns:
            Total sum of control area sizes
        """
        if adj is None:
            adj = self.adj
        if adj is None:
            raise ValueError("No adjacency matrix provided")

        total = 0
        for nd in nodes:
            total += int(adj[nd].sum()) + 1  # neighbors + self
        return total

    # ================================================================
    # 5. Community Detection
    # ================================================================

    def community_detection(self, method: str = 'greedy',
                          adj: Optional[np.ndarray] = None,
                          use_cache: bool = True) -> np.ndarray:
        """
        Detect communities in undirected binary network.

        Args:
            method: 'greedy' (default) — greedy modularity maximization
            adj: Optional adjacency matrix (uses self.adj if None)
            use_cache: If True, cache result for repeated calls

        Returns:
            (N,) array of community labels (0-indexed)
        """
        if adj is None:
            adj = self.adj
        if adj is None:
            raise ValueError("No adjacency matrix provided")

        # Return cached result if available
        if use_cache and self._communities_cache is not None:
            return self._communities_cache

        if method == 'greedy':
            communities = self._greedy_modularity(adj)
        else:
            raise ValueError(f"Unknown community method: {method}")

        # Cache if requested
        if use_cache:
            self._communities_cache = communities

        return communities

    def _greedy_modularity(self, adj: np.ndarray) -> np.ndarray:
        """
        Greedy agglomerative modularity maximization.

        Uses NetworkX if available for better performance,
        falls back to simple label propagation otherwise.

        Args:
            adj: Binary adjacency matrix

        Returns:
            (N,) array of community labels
        """
        try:
            import networkx as nx
            from networkx.algorithms.community import greedy_modularity_communities

            G = nx.from_numpy_array(adj)
            communities = greedy_modularity_communities(G)

            labels = np.zeros(adj.shape[0], dtype=np.int32)
            for ci, comm in enumerate(communities):
                for node in comm:
                    labels[node] = ci
            return labels

        except ImportError:
            # Fallback: simple label propagation
            logger.debug("NetworkX not available, using label propagation fallback")
            return self._label_propagation(adj)

    def _label_propagation(self, adj: np.ndarray) -> np.ndarray:
        """
        Simple label propagation community detection (fallback).

        Args:
            adj: Binary adjacency matrix

        Returns:
            (N,) array of community labels
        """
        n = adj.shape[0]
        labels = np.arange(n, dtype=np.int32)
        changed = True
        max_iter = 100
        it = 0

        while changed and it < max_iter:
            changed = False
            order = np.random.permutation(n)

            for v in order:
                neighbors = np.where(adj[v] > 0)[0]
                if len(neighbors) == 0:
                    continue

                neighbor_labels = labels[neighbors]
                unique, counts = np.unique(neighbor_labels, return_counts=True)
                best = unique[np.argmax(counts)]

                if best != labels[v]:
                    labels[v] = best
                    changed = True

            it += 1

        # Re-label to 0..K-1
        unique_labels = np.unique(labels)
        remap = {old: new for new, old in enumerate(unique_labels)}
        return np.array([remap[l] for l in labels], dtype=np.int32)

    # ================================================================
    # 6. Participation Coefficient
    # ================================================================

    def participation_coefficient(self, communities: np.ndarray,
                                 adj: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Participation coefficient for each node.

        PC_i = 1 - Σ_s (k_is / k_i)^2

        where k_is = number of edges from node i to community s,
              k_i  = total degree of node i.

        PC ≈ 0: mostly intra-module connections (provincial)
        PC ≈ 1: evenly distributed across modules (connector)

        Args:
            communities: (N,) array of community labels
            adj: Optional adjacency matrix (uses self.adj if None)

        Returns:
            (N,) array of participation coefficients
        """
        if adj is None:
            adj = self.adj
        if adj is None:
            raise ValueError("No adjacency matrix provided")

        n = adj.shape[0]
        deg = adj.sum(axis=1).astype(np.float64)
        pc = np.zeros(n, dtype=np.float64)

        unique_comm = np.unique(communities)

        for i in range(n):
            if deg[i] == 0:
                pc[i] = 0.0
                continue

            neighbors = np.where(adj[i] > 0)[0]
            neighbor_comm = communities[neighbors]

            s = 0.0
            for c in unique_comm:
                k_is = np.sum(neighbor_comm == c)
                s += (k_is / deg[i]) ** 2

            pc[i] = 1.0 - s

        return pc


# ================================================================
# Standalone Functions (Backward Compatibility)
# ================================================================

def proportional_threshold(conn: np.ndarray, density: float) -> np.ndarray:
    """Standalone wrapper for NetworkAnalyzer.proportional_threshold()."""
    return NetworkAnalyzer.proportional_threshold(conn, density)


def degrees(adj: np.ndarray) -> np.ndarray:
    """Standalone wrapper for NetworkAnalyzer.degrees()."""
    analyzer = NetworkAnalyzer(adj)
    return analyzer.degrees()


def top_degree_nodes(adj: np.ndarray, frac: float) -> np.ndarray:
    """Standalone wrapper for NetworkAnalyzer.top_degree_nodes()."""
    analyzer = NetworkAnalyzer(adj)
    return analyzer.top_degree_nodes(frac)


def find_mdset(adj: np.ndarray) -> np.ndarray:
    """Standalone wrapper for NetworkAnalyzer.find_mdset()."""
    analyzer = NetworkAnalyzer(adj)
    return analyzer.find_mdset(use_cache=False)


def control_area(adj: np.ndarray, node: int) -> np.ndarray:
    """Standalone wrapper for NetworkAnalyzer.control_area()."""
    analyzer = NetworkAnalyzer(adj)
    return analyzer.control_area(node)


def union_control_area(adj: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Standalone wrapper for NetworkAnalyzer.union_control_area()."""
    analyzer = NetworkAnalyzer(adj)
    return analyzer.union_control_area(nodes)


def sum_control_area_sizes(adj: np.ndarray, nodes: np.ndarray) -> int:
    """Standalone wrapper for NetworkAnalyzer.sum_control_area_sizes()."""
    analyzer = NetworkAnalyzer(adj)
    return analyzer.sum_control_area_sizes(nodes)


def community_detection(adj: np.ndarray, method: str = 'greedy') -> np.ndarray:
    """Standalone wrapper for NetworkAnalyzer.community_detection()."""
    analyzer = NetworkAnalyzer(adj)
    return analyzer.community_detection(method, use_cache=False)


def participation_coefficient(adj: np.ndarray, communities: np.ndarray) -> np.ndarray:
    """Standalone wrapper for NetworkAnalyzer.participation_coefficient()."""
    analyzer = NetworkAnalyzer(adj)
    return analyzer.participation_coefficient(communities)
