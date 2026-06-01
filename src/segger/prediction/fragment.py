"""Fragment mode for grouping unassigned transcripts.

Builds a spatial k-NN graph, weights edges by transcript-embedding cosine,
splits oversized connected components, and merges compatible adjacent
communities. RAPIDS is used when available; scipy/numpy provide the CPU
fallback used by tests.
"""

from dataclasses import dataclass
import numpy as np

try:
    import cupy as cp
    import cudf
    import cugraph
    import cuml
    from cupyx.scipy.sparse import coo_matrix as cp_coo_matrix
    from cupyx.scipy.sparse.csgraph import connected_components as cc_gpu
    HAS_RAPIDS = True
except ImportError:
    HAS_RAPIDS = False

from scipy.spatial import cKDTree as KDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as cc_cpu


@dataclass
class FragmentConfig:
    """Hyperparameters for fragment mode."""
    min_transcripts: int = 50
    max_transcripts: int = 5000
    n_neighbors: int = 15
    edge_threshold: float = 0.0   # drop k-NN edges with embedding cosine below this
    resolution: float = 1.0       # Leiden resolution (higher -> smaller communities)
    merge_threshold: float = 0.6  # min mean-embedding cosine to merge adjacent communities
    use_gpu: bool = True

    def __post_init__(self):
        if self.min_transcripts <= 0:
            raise ValueError("min_transcripts must be positive.")
        if self.max_transcripts < self.min_transcripts:
            raise ValueError("max_transcripts must be >= min_transcripts.")
        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive.")
        if not -1.0 <= self.edge_threshold <= 1.0:
            raise ValueError("edge_threshold must be in [-1, 1].")
        if self.resolution <= 0:
            raise ValueError("resolution must be positive.")
        if not -1.0 <= self.merge_threshold <= 1.0:
            raise ValueError("merge_threshold must be in [-1, 1].")


def _build_graph(xy: np.ndarray, emb: np.ndarray, config: FragmentConfig, use_gpu: bool):
    """Spatial k-NN graph, embedding-weighted + pruned, then connected components.

    Returns host arrays ``(src, dst, weight, dist, labels)`` where ``labels`` are
    connected-component ids over the ``n`` transcripts.
    """
    n = xy.shape[0]
    k = min(config.n_neighbors + 1, n)  # +1: self is included

    if use_gpu:
        xy_d = cp.asarray(xy, dtype=cp.float32)
        emb_d = cp.asarray(emb, dtype=cp.float32)
        nn = cuml.neighbors.NearestNeighbors(n_neighbors=k)
        nn.fit(xy_d)
        dist, idx = nn.kneighbors(xy_d)
        src = cp.repeat(cp.arange(n, dtype=cp.int32), k)
        dst = idx.reshape(-1).astype(cp.int32)
        d = dist.reshape(-1).astype(cp.float32)
        cos = cp.einsum("ij,ij->i", emb_d[src], emb_d[dst])
        keep = (src != dst) & (cos >= config.edge_threshold)
        src, dst, d, cos = src[keep], dst[keep], d[keep], cos[keep]
        data = cp.ones(src.size, dtype=cp.float32)
        adj = cp_coo_matrix((data, (src, dst)), shape=(n, n)).tocsr()
        _, labels = cc_gpu(adj, directed=False)
        return (
            cp.asnumpy(src).astype(np.int64),
            cp.asnumpy(dst).astype(np.int64),
            cp.asnumpy(0.5 * (cos + 1.0)),
            cp.asnumpy(d),
            cp.asnumpy(labels),
        )

    tree = KDTree(xy)
    dist, idx = tree.query(xy, k=k)
    dist = np.atleast_2d(dist)
    idx = np.atleast_2d(idx)
    src = np.repeat(np.arange(n, dtype=np.int64), idx.shape[1])
    dst = idx.reshape(-1).astype(np.int64)
    d = dist.reshape(-1).astype(np.float32)
    cos = np.einsum("ij,ij->i", emb[src], emb[dst])
    keep = (src != dst) & (cos >= config.edge_threshold)
    src, dst, d, cos = src[keep], dst[keep], d[keep], cos[keep]
    data = np.ones(src.size, dtype=np.float32)
    adj = coo_matrix((data, (src, dst)), shape=(n, n)).tocsr()
    _, labels = cc_cpu(adj, directed=False)
    return src, dst, (0.5 * (cos + 1.0)).astype(np.float32), d, labels.astype(np.int64)


def _leiden_partition(n_local, s, t, w, resolution) -> np.ndarray:
    """Partition a local graph (vertices ``0..n_local-1``) into communities (GPU)."""
    gdf = cudf.DataFrame({
        "src": cp.asarray(s, dtype=cp.int32),
        "dst": cp.asarray(t, dtype=cp.int32),
        "w": cp.asarray(w, dtype=cp.float32),
    })
    G = cugraph.Graph()
    G.from_cudf_edgelist(
        gdf,
        source="src",
        destination="dst",
        edge_attr="w",
        renumber=True,
    )
    try:
        parts, _ = cugraph.leiden(G, resolution=resolution, random_state=0)
    except (AttributeError, TypeError):
        parts, _ = cugraph.louvain(G, resolution=resolution)
    verts = parts["vertex"].to_numpy()
    prt = parts["partition"].to_numpy()
    lab = np.full(n_local, -1, dtype=np.int64)
    lab[verts] = prt
    # Isolated vertices (no surviving edges) get their own singleton community.
    iso = lab < 0
    if iso.any():
        lab[iso] = (lab.max() + 1) + np.arange(int(iso.sum()))
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def _threshold_cut(n_local, s, t, w, d) -> np.ndarray:
    """CPU split: drop weakest (low-cos, then long) edges until the graph splits."""
    m = w.size
    if m == 0:
        return np.arange(n_local)
    order = np.lexsort((-d, w))  # primary: w ascending; ties: distance descending
    for frac in (0.1, 0.2, 0.3, 0.5, 0.8, 1.0):
        cut = int(m * frac)
        keep = np.ones(m, dtype=bool)
        keep[order[:cut]] = False
        adj = coo_matrix(
            (np.ones(int(keep.sum()), np.float32), (s[keep], t[keep])),
            shape=(n_local, n_local),
        ).tocsr()
        ncc, lab = cc_cpu(adj, directed=False)
        if ncc > 1:
            return lab.astype(np.int64)
    return np.arange(n_local)


def _split_component(nodes, s, t, w, d, config, use_gpu) -> np.ndarray:
    """Split one oversized component so every piece <= ``max_transcripts``.

    ``s``/``t`` are edges (global node ids) internal to the component. Returns
    sub-community ids aligned with ``nodes``.
    """
    pos = np.full(int(nodes.max()) + 1, -1, dtype=np.int64)
    pos[nodes] = np.arange(nodes.size)
    ls, lt = pos[s], pos[t]

    out = np.full(nodes.size, -1, dtype=np.int64)
    next_id = 0
    res_cap = config.resolution * 32.0
    stack = [(np.arange(nodes.size), ls, lt, w, d, config.resolution)]
    while stack:
        ni, es, et, ew, ed, res = stack.pop()
        if ni.size <= config.max_transcripts:
            out[ni] = next_id
            next_id += 1
            continue
        if use_gpu:
            parts = _leiden_partition(nodes.size, es, et, ew, res)[ni]
        else:
            local = np.full(nodes.size, -1, dtype=np.int64)
            local[ni] = np.arange(ni.size)
            parts = _threshold_cut(ni.size, local[es], local[et], ew, ed)
        uniq = np.unique(parts)
        if uniq.size <= 1:
            if res < res_cap and use_gpu:
                stack.append((ni, es, et, ew, ed, res * 2.0))
            else:
                for start in range(0, ni.size, config.max_transcripts):
                    end = start + config.max_transcripts
                    out[ni[start:end]] = next_id
                    next_id += 1
            continue
        pnode = np.full(nodes.size, -1, dtype=np.int64)
        pnode[ni] = parts
        es_p, et_p = pnode[es], pnode[et]
        for p in uniq:
            em = (es_p == p) & (et_p == p)
            stack.append((
                ni[parts == p],
                es[em],
                et[em],
                ew[em],
                ed[em],
                config.resolution,
            ))
    return out


def _enforce_max(labels, src, dst, weight, dist, config, use_gpu) -> np.ndarray:
    """Split every component above ``max_transcripts`` via Leiden / threshold cut."""
    counts = np.bincount(labels)
    oversized = np.nonzero(counts > config.max_transcripts)[0]
    if oversized.size == 0:
        return labels
    out = labels.copy()
    next_id = int(labels.max()) + 1
    lab_src, lab_dst = labels[src], labels[dst]
    internal = lab_src == lab_dst
    for comp in oversized:
        nodes = np.nonzero(labels == comp)[0]
        em = internal & (lab_src == comp)
        sub = _split_component(nodes, src[em], dst[em], weight[em], dist[em], config, use_gpu)
        for sid in np.unique(sub):
            out[nodes[sub == sid]] = next_id
            next_id += 1
    return out


def _community_sums(labels, emb, K) -> np.ndarray:
    """Per-community embedding sums via per-dimension bincount (C-level, fast)."""
    sums = np.empty((K, emb.shape[1]), dtype=np.float64)
    for c in range(emb.shape[1]):
        sums[:, c] = np.bincount(labels, weights=emb[:, c], minlength=K)
    return sums


def _flatten(parent: np.ndarray) -> np.ndarray:
    """Vectorised union-find: collapse every node to its root by pointer jumping."""
    while True:
        nxt = parent[parent]
        if np.array_equal(nxt, parent):
            return parent
        parent = nxt


def _merge_and_finalize(labels, src, dst, emb, config) -> np.ndarray:
    """Vectorised mutual-best-neighbour region merge, then drop sub-min communities.

    Each round every community keeps a single best admissible neighbour (highest
    mean-embedding cosine, combined size <= max, ties broken by edge id); mutually
    best pairs form a matching and are merged. Bounded rounds, no per-community
    Python loop -- so it stays light at slide scale.
    """
    _, labels = np.unique(labels, return_inverse=True)
    K = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=K).astype(np.float64)
    sums = _community_sums(labels, emb, K)

    # Undirected, de-duplicated community-adjacency pairs.
    cs, cd = labels[src], labels[dst]
    m = cs != cd
    if m.any():
        a = np.minimum(cs[m], cd[m]).astype(np.int64)
        b = np.maximum(cs[m], cd[m]).astype(np.int64)
        pk = np.unique(a * np.int64(K) + b)
        pa, pb = pk // K, pk % K
    else:
        pa = pb = np.empty(0, dtype=np.int64)

    parent = np.arange(K, dtype=np.int64)
    for _ in range(64):  # safety bound; merges roughly halve live roots per round
        if pa.size == 0:
            break
        root = _flatten(parent)
        ra, rb = root[pa], root[pb]
        live = ra != rb
        if not live.any():
            break
        ra, rb = ra[live], rb[live]
        rsize = np.bincount(root, weights=counts, minlength=K)
        rsum = np.empty_like(sums)
        for c in range(sums.shape[1]):
            rsum[:, c] = np.bincount(root, weights=sums[:, c], minlength=K)
        rmean = rsum / np.maximum(rsize, 1.0)[:, None]
        rmean /= np.linalg.norm(rmean, axis=1, keepdims=True) + 1e-12
        cos = np.einsum("ij,ij->i", rmean[ra], rmean[rb])
        ok = (cos >= config.merge_threshold) & (rsize[ra] + rsize[rb] <= config.max_transcripts)
        if not ok.any():
            break
        ra, rb, cos = ra[ok], rb[ok], cos[ok]
        # Each root's unique best edge: highest cos, ties broken by smallest id.
        eid = np.arange(ra.size)
        endp = np.concatenate([ra, rb])
        ecos = np.concatenate([cos, cos])
        eidx = np.concatenate([eid, eid])
        best_cos = np.full(K, -np.inf)
        np.maximum.at(best_cos, endp, ecos)
        at_best = ecos >= best_cos[endp] - 1e-9
        chosen = np.full(K, ra.size, dtype=np.int64)  # sentinel = no edge
        np.minimum.at(chosen, endp[at_best], eidx[at_best])
        mutual = (chosen[ra] == eid) & (chosen[rb] == eid)  # a matching
        if not mutual.any():
            break
        parent[np.maximum(ra[mutual], rb[mutual])] = np.minimum(ra[mutual], rb[mutual])

    root = _flatten(parent)
    final = root[labels]
    rsize = np.bincount(final, minlength=K)
    out = np.full(labels.shape[0], -1, dtype=np.int64)
    keep = rsize[final] >= config.min_transcripts
    if keep.any():
        _, out_ids = np.unique(final[keep], return_inverse=True)
        out[keep] = out_ids
    return out


def assign_fragments(
    xy: np.ndarray,
    emb: np.ndarray,
    config: FragmentConfig | None = None,
) -> np.ndarray:
    """Assign fragment IDs to a set of unassigned transcripts.

    Parameters
    ----------
    xy : np.ndarray
        ``(N, 2)`` transcript coordinates.
    emb : np.ndarray
        ``(N, D)`` GNN transcript embeddings (L2-normalised internally).
    config : FragmentConfig, optional
        Hyperparameters; defaults to ``FragmentConfig()``.

    Returns
    -------
    np.ndarray
        ``(N,)`` fragment IDs (``-1`` where the transcript is not part of a
        fragment within ``[min_transcripts, max_transcripts]``).
    """
    if config is None:
        config = FragmentConfig()
    n = xy.shape[0]
    out = np.full(n, -1, dtype=np.int64)
    if n == 0 or n < config.min_transcripts:
        return out

    use_gpu = config.use_gpu and HAS_RAPIDS
    xy = np.ascontiguousarray(xy, dtype=np.float32)
    emb = np.ascontiguousarray(emb, dtype=np.float32)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    src, dst, weight, dist, labels = _build_graph(xy, emb, config, use_gpu)
    if src.size == 0:
        return out

    labels = _enforce_max(labels, src, dst, weight, dist, config, use_gpu)
    return _merge_and_finalize(labels, src, dst, emb, config)
