"""Optional post-hoc refinement of the per-point label votes.

  - smooth : vote diffusion on a spatial kNN graph (numpy/scipy only)
  - crf    : multi-label graph cut, Potts or per-category compatibility
             matrix (requires pygco)

Both operate on the existing votes only: points with no votes keep label -1,
so coverage is unchanged and refine on/off comparisons are apples-to-apples
(only the labels of already-voted points can change, not how many points are
classified). The paper's headline results use no refinement.
"""
import numpy as np
from scipy.spatial import cKDTree


# Compatibility matrices for the CRF pairwise term.
# pairwise[c, c'] = cost of a boundary between neighbouring points labelled c, c':
#   diagonal = 0 (same part, no penalty)
#   parts that are physically adjacent  -> low cost (~Potts)
#   parts that never touch              -> high cost (implausible boundary)
# Label order follows CATEGORY_LABELS for the category.
COMPAT_MATRICES = {
    # 04379243 Table: [top, leg, shelf]
    # Legs are adjacent to both top and shelf; top and shelf never touch directly.
    '04379243': np.array([
        [0.0, 1.0, 2.5],
        [1.0, 0.0, 1.0],
        [2.5, 1.0, 0.0],
    ], dtype=float),
}


def get_compat(category_id: str):
    """Return the compatibility matrix for a category, or None (pure Potts)."""
    return COMPAT_MATRICES.get(category_id)


def build_votes_matrix(point_votes: dict, n_points: int, prompts: list) -> np.ndarray:
    """Convert {point_id: {label: count}} into an N x C vote matrix."""
    C = len(prompts)
    votes = np.zeros((n_points, C), dtype=np.float64)
    idx_of = {p: i for i, p in enumerate(prompts)}
    for pid, vd in point_votes.items():
        for lab, cnt in vd.items():
            if lab in idx_of:
                votes[pid, idx_of[lab]] += cnt
    return votes


def _knn_graph(points: np.ndarray, tree: cKDTree, k: int):
    dists, idx = tree.query(points, k=k + 1)  # column 0 is the point itself
    return dists[:, 1:], idx[:, 1:]


def refine_smooth(points, votes, tree=None, k=8, sigma=None, alpha=0.5, n_iter=10):
    """Label propagation: votes diffuse across spatial neighbours.

    alpha weights the neighbourhood contribution, n_iter is the number of
    iterations, sigma is the kernel scale (median neighbour distance if None).
    """
    if tree is None:
        tree = cKDTree(points)
    dists, idx = _knn_graph(points, tree, k)
    if sigma is None:
        sigma = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
    w = np.exp(-(dists ** 2) / (2.0 * sigma ** 2))            # N x k
    w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
    P0 = votes / (votes.sum(axis=1, keepdims=True) + 1e-12)   # N x C (0 where no votes)
    P = P0.copy()
    for _ in range(n_iter):
        neigh = np.einsum('nk,nkc->nc', w, P[idx])            # weighted neighbour average
        P = (1.0 - alpha) * P0 + alpha * neigh                # anchored to original votes
    return P.argmax(axis=1)


def refine_crf(points, votes, tree=None, k=8, sigma=None, lam=1.0, compat=None):
    """Graph cut: unary = -log(votes), pairwise = Potts (compat=None) or a C x C matrix."""
    import pygco
    if tree is None:
        tree = cKDTree(points)
    dists, idx = _knn_graph(points, tree, k)
    if sigma is None:
        sigma = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
    n, C = votes.shape
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    wts = np.exp(-(dists.ravel() ** 2) / (2.0 * sigma ** 2))
    m = rows < cols                                           # unique undirected edges
    edges = np.stack([rows[m], cols[m]], axis=1).astype(np.int32)
    edge_w = (lam * wts[m]).astype(np.float64)
    P = votes / (votes.sum(axis=1, keepdims=True) + 1e-12)
    unary = (-np.log(P + 1e-6)).astype(np.float64)            # N x C
    pairwise = (1.0 - np.eye(C)) if compat is None else np.asarray(compat, dtype=float)
    out = pygco.cut_general_graph(edges, edge_w, unary,
                                  pairwise.astype(np.float64), algorithm='expansion')
    return np.asarray(out)


def refine_labels(points, point_votes, n_points, prompts, refine="none",
                  tree=None, k=8, sigma=None, alpha=0.5, n_iter=10, lam=1.0, compat=None):
    """Entry point. Returns per-point labels (-1 for unvoted points, coverage unchanged)."""
    votes = build_votes_matrix(point_votes, n_points, prompts)
    has_vote = votes.sum(axis=1) > 0
    labels = np.full(n_points, -1, dtype=int)
    if refine == "none" or not has_vote.any():
        labels[has_vote] = votes[has_vote].argmax(axis=1)
        return labels
    if refine == "smooth":
        ref = refine_smooth(points, votes, tree, k, sigma, alpha, n_iter)
    elif refine == "crf":
        ref = refine_crf(points, votes, tree, k, sigma, lam, compat)
    else:
        raise ValueError(f"Unknown refine mode: {refine}")
    labels[has_vote] = ref[has_vote]
    return labels
