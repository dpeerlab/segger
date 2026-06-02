"""Fragment mode for grouping unassigned (orphan) transcripts into de-novo cells.

This is **Stage B** of the additive, unassigned-only recovery pipeline. It only ever sees
transcripts that primary segmentation (and Stage A extension) left unassigned; the
orchestration / coalescing lives in ``prediction/recovery.py`` and ``data/writer.py``.

Why the design here is *anti-round*
------------------------------------
The original implementation (symmetric k-NN + connected components + modularity Leiden +
global mean-embedding merge) is structurally biased toward **round** cells: symmetric k-NN
fills the "waist" of a thin neurite with cross-chain edges, and a global mean-embedding
merge pulls drifting chain segments back toward a centroid blob. Fragments exist to recover
**elongated / complex** morphologies (neurons, glia, endothelial processes), so round output
defeats the purpose.

Two selectable backends (``config.method``):

``leiden`` (default, conservative, lowest churn)
    1. **Mutual** k-NN over orphan ``xy``: keep edge ``(i, j)`` only if ``i`` is in
       ``kNN(j)`` *and* ``j`` is in ``kNN(i)``. In an isotropic blob mutual ≈ symmetric, but
       along a thin neurite the cross-chain "waist" neighbours are asymmetric and dropped, so
       the graph follows the 1-D chain instead of filling it. This is THE anti-roundness move.
    2. Edge weight ``w = 0.5 * (cos + 1)`` in ``[0, 1]``; prune edges whose *raw* cosine is
       below ``edge_threshold`` so spatially-adjacent but expression-discordant pairs never
       seed a component.
    3. Connected components on the pruned mutual graph -> seed components.
    4. Split **only** components above ``max_transcripts`` via weighted Leiden
       (cuGraph GPU / leidenalg+igraph CPU; ``_threshold_cut`` numpy as last resort), recursing
       until each piece <= cap. Components already <= cap are **never** force-split
       (splitting a small chain re-rounds it).
    5. **Contact-interface chain-only merge**: build the community region-adjacency graph from
       surviving inter-community mutual-kNN edges; score each adjacent pair by the mean
       embedding cosine over ONLY the boundary transcripts forming its edges (the contact
       interface), not centroids. Merge mutual-best adjacent pairs iff
       ``contact_cos >= merge_threshold`` and the union stays <= cap; iterate to a fixed point.
       Two segments of one neurite (drifting centroids but agreeing contact tx) merge
       end-to-end -> elongated; two unlike cells touching at a thin low-cosine interface do not.
    6. Communities below ``min_transcripts`` -> noise (``-1``).

``hdbscan`` (grafted bake-off backend, variable-density / arbitrary shape)
    Cluster the co-scaled joint matrix ``F = [xy / space_scale,
    emb_unit * emb_weight * sqrt(2 / D)]`` with cuML (GPU) / sklearn (CPU) HDBSCAN. No size-cap
    split (the round-forcer); ``max_transcripts`` is QC/log-only here.

RAPIDS is used when available; scipy / scikit-learn / leidenalg provide the CPU fallback used
by tests.
"""

from dataclasses import dataclass
import math
import warnings

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

# CPU Leiden parity (soft optional dependency, `cluster` extra). Guarded so the package
# imports without it; falls back to the numpy `_threshold_cut`.
try:
    import igraph as _ig
    import leidenalg as _la

    HAS_LEIDENALG = True
except ImportError:
    HAS_LEIDENALG = False

from scipy.spatial import cKDTree as KDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as cc_cpu


@dataclass
class FragmentConfig:
    """Hyperparameters for fragment mode (Stage B)."""

    min_transcripts: int = 50
    max_transcripts: int = 5000
    n_neighbors: int = 15
    # Drop k-NN edges whose *raw* embedding cosine is below this (RAISED 0.0 -> 0.30
    # so spatially-adjacent but expression-discordant pairs never seed a component).
    edge_threshold: float = 0.30
    resolution: float = 1.0       # Leiden resolution (higher -> smaller communities)
    merge_threshold: float = 0.6  # min contact-interface cosine to merge adjacent communities
    use_gpu: bool = True

    # --- backend selection ---
    # 'quickshift' (DEFAULT): embedding-density mode-seeking forest -- fast, deterministic,
    #     follows filaments/neurites, no resolution/density-tree. See _quickshift_cluster.
    # 'leiden' / 'hdbscan': retained alternates (bake-off / ablation).
    method: str = "quickshift"
    mutual_knn: bool = True       # leiden: keep only mutual (reciprocal) k-NN edges
    emb_weight: float = 1.0       # HDBSCAN: weight of the embedding block in joint matrix
    space_scale: float = 5.0      # HDBSCAN: ~half median nuclear radius (um) for xy scaling
    min_samples: int | None = None  # HDBSCAN min_samples; default max(5, min_transcripts//4)
    cluster_selection: str = "eom"  # HDBSCAN cluster_selection_method: 'eom' | 'leaf'

    # --- quickshift ---
    # Spatial reach for an admissible parent link, as a multiple of the median nearest-neighbour
    # distance (adapts to local density). A point only links to a higher-density neighbour within
    # this reach AND with embedding cosine >= edge_threshold, so links never cross a cell seam.
    quickshift_max_dist_factor: float = 3.0
    quickshift_max_dist: float | None = None  # optional hard spatial cap (um); None -> off
    # Persistence (ToMATo) merge: adjacent density basins merge unless the shallower basin's
    # prominence (peak - saddle) exceeds this fraction of the global peak. Higher -> fewer, larger
    # fragments (a uniform cell stays whole); lower -> split same-type touching cells at the valley.
    quickshift_persistence: float = 0.5

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
        if self.method not in {"quickshift", "leiden", "hdbscan"}:
            raise ValueError("method must be 'quickshift', 'leiden', or 'hdbscan'.")
        if self.quickshift_max_dist_factor <= 0:
            raise ValueError("quickshift_max_dist_factor must be positive.")
        if self.quickshift_max_dist is not None and self.quickshift_max_dist <= 0:
            raise ValueError("quickshift_max_dist must be positive when set.")
        if not 0.0 <= self.quickshift_persistence <= 1.0:
            raise ValueError("quickshift_persistence must be in [0, 1].")
        if self.cluster_selection not in {"eom", "leaf"}:
            raise ValueError("cluster_selection must be 'eom' or 'leaf'.")
        if self.emb_weight < 0:
            raise ValueError("emb_weight must be non-negative.")
        if self.space_scale <= 0:
            raise ValueError("space_scale must be positive.")
        if self.min_samples is not None and self.min_samples <= 0:
            raise ValueError("min_samples must be positive when set.")


# --------------------------------------------------------------------------------------
# Graph construction (mutual k-NN, embedding-weighted, pruned) + connected components
# --------------------------------------------------------------------------------------
def _build_graph(xy: np.ndarray, emb: np.ndarray, config: FragmentConfig, use_gpu: bool):
    """Spatial k-NN graph, embedding-weighted + pruned + (optionally) mutualised.

    Returns host arrays ``(src, dst, weight, dist, labels)`` where ``labels`` are
    connected-component ids over the ``n`` transcripts. ``(src, dst)`` is symmetric
    (each surviving undirected edge appears in both directions) so downstream
    community-adjacency logic sees both endpoints.
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
        keep_self = src != dst
        src, dst, d = src[keep_self], dst[keep_self], d[keep_self]
        cos = cp.einsum("ij,ij->i", emb_d[src], emb_d[dst])
        keep = cos >= config.edge_threshold
        src, dst, d, cos = src[keep], dst[keep], d[keep], cos[keep]
        src_h = cp.asnumpy(src).astype(np.int64)
        dst_h = cp.asnumpy(dst).astype(np.int64)
        d_h = cp.asnumpy(d).astype(np.float32)
        cos_h = cp.asnumpy(cos).astype(np.float64)
    else:
        tree = KDTree(xy)
        dist, idx = tree.query(xy, k=k)
        dist = np.atleast_2d(dist)
        idx = np.atleast_2d(idx)
        src = np.repeat(np.arange(n, dtype=np.int64), idx.shape[1])
        dst = idx.reshape(-1).astype(np.int64)
        d = dist.reshape(-1).astype(np.float32)
        keep_self = src != dst
        src, dst, d = src[keep_self], dst[keep_self], d[keep_self]
        cos = np.einsum("ij,ij->i", emb[src], emb[dst])
        keep = cos >= config.edge_threshold
        src_h, dst_h, d_h, cos_h = src[keep], dst[keep], d[keep], cos[keep].astype(np.float64)

    # Symmetrise into undirected edges. With mutual_knn we keep an undirected edge only when
    # BOTH directed edges survived (reciprocal); otherwise we keep the union (either direction).
    src_h, dst_h, d_h, cos_h = _symmetrise(
        src_h, dst_h, d_h, cos_h, n, mutual=config.mutual_knn
    )
    if src_h.size == 0:
        return (
            np.empty(0, np.int64),
            np.empty(0, np.int64),
            np.empty(0, np.float32),
            np.empty(0, np.float32),
            np.arange(n, dtype=np.int64),
        )

    weight = (0.5 * (cos_h + 1.0)).astype(np.float32)
    adj = coo_matrix((np.ones(src_h.size, np.float32), (src_h, dst_h)), shape=(n, n)).tocsr()
    _, labels = cc_cpu(adj, directed=False)
    return src_h, dst_h, weight, d_h, labels.astype(np.int64)


def _symmetrise(src, dst, d, cos, n, *, mutual: bool):
    """Turn a directed edge list into a symmetric undirected one (both directions emitted).

    With ``mutual=True``, an undirected pair survives only if the directed edge was present in
    BOTH directions (reciprocal k-NN). With ``mutual=False`` the union is kept. Per-edge
    distance/cosine are deterministic (they are symmetric functions of the endpoints, so the
    two directions carry identical values; we take the first occurrence).
    """
    if src.size == 0:
        return src, dst, d, cos
    a = np.minimum(src, dst).astype(np.int64)
    b = np.maximum(src, dst).astype(np.int64)
    key = a * np.int64(n) + b
    if mutual:
        # A reciprocal pair contributes the key twice (i->j and j->i); keep keys seen >= 2.
        uniq, counts = np.unique(key, return_counts=True)
        recip = uniq[counts >= 2]
        sel = np.isin(key, recip)
        a, b, key = a[sel], b[sel], key[sel]
        d, cos = d[sel], cos[sel]
    # Deduplicate undirected pairs (first occurrence wins; values are symmetric).
    order = np.argsort(key, kind="stable")
    key_s, a_s, b_s = key[order], a[order], b[order]
    d_s, cos_s = d[order], cos[order]
    first = np.ones(key_s.size, dtype=bool)
    first[1:] = key_s[1:] != key_s[:-1]
    a_u, b_u, d_u, cos_u = a_s[first], b_s[first], d_s[first], cos_s[first]
    # Emit both directions for an undirected graph.
    su = np.concatenate([a_u, b_u])
    du = np.concatenate([b_u, a_u])
    du_d = np.concatenate([d_u, d_u])
    du_cos = np.concatenate([cos_u, cos_u])
    return su.astype(np.int64), du.astype(np.int64), du_d.astype(np.float32), du_cos


# --------------------------------------------------------------------------------------
# Oversized-component splitting (Leiden GPU / leidenalg CPU / threshold-cut fallback)
# --------------------------------------------------------------------------------------
def _leiden_partition_gpu(n_local, s, t, w, resolution) -> np.ndarray:
    """Partition a local graph (vertices ``0..n_local-1``) into communities (cuGraph)."""
    gdf = cudf.DataFrame(
        {
            "src": cp.asarray(s, dtype=cp.int32),
            "dst": cp.asarray(t, dtype=cp.int32),
            "w": cp.asarray(w, dtype=cp.float32),
        }
    )
    G = cugraph.Graph()
    G.from_cudf_edgelist(gdf, source="src", destination="dst", edge_attr="w", renumber=True)
    try:
        parts, _ = cugraph.leiden(G, resolution=resolution, random_state=0)
    except (AttributeError, TypeError):
        parts, _ = cugraph.louvain(G, resolution=resolution)
    verts = parts["vertex"].to_numpy()
    prt = parts["partition"].to_numpy()
    lab = np.full(n_local, -1, dtype=np.int64)
    lab[verts] = prt
    iso = lab < 0
    if iso.any():
        lab[iso] = (lab.max() + 1) + np.arange(int(iso.sum()))
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def _leiden_partition_cpu(n_local, s, t, w, resolution) -> np.ndarray:
    """Partition a local graph into communities with leidenalg (RBConfigurationVertexPartition).

    Mirrors cugraph.leiden semantics (modularity-style with a resolution parameter) so GPU/CPU
    paths agree on partition structure. Deterministic via a fixed seed.
    """
    edges = list(zip(s.tolist(), t.tolist()))
    g = _ig.Graph(n=int(n_local), edges=edges, directed=False)
    g.es["weight"] = w.astype(float).tolist()
    part = _la.find_partition(
        g,
        _la.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(resolution),
        seed=0,
    )
    lab = np.asarray(part.membership, dtype=np.int64)
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def _threshold_cut(n_local, s, t, w, d) -> np.ndarray:
    """Last-resort CPU split: drop weakest (low-cos, then long) edges until the graph splits."""
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


def _partition_local(n_local, s, t, w, d, resolution, use_gpu) -> np.ndarray:
    """Dispatch a single Leiden split: cuGraph (GPU) -> leidenalg (CPU) -> threshold-cut."""
    if use_gpu:
        return _leiden_partition_gpu(n_local, s, t, w, resolution)
    if HAS_LEIDENALG:
        return _leiden_partition_cpu(n_local, s, t, w, resolution)
    return _threshold_cut(n_local, s, t, w, d)


def _split_component(nodes, s, t, w, d, config, use_gpu) -> np.ndarray:
    """Split one oversized component so every piece <= ``max_transcripts``.

    ``s``/``t`` are edges (global node ids) internal to the component. Returns sub-community ids
    aligned with ``nodes``. Recurses with increasing resolution; the numeric ``_threshold_cut``
    fallback (or a forced index split) guarantees termination when a community cannot be
    partitioned any further.
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
        local = np.full(nodes.size, -1, dtype=np.int64)
        local[ni] = np.arange(ni.size)
        parts = _partition_local(ni.size, local[es], local[et], ew, ed, res, use_gpu)
        uniq = np.unique(parts)
        if uniq.size <= 1:
            if res < res_cap:
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
            stack.append((ni[parts == p], es[em], et[em], ew[em], ed[em], config.resolution))
    return out


def _enforce_max(labels, src, dst, weight, dist, config, use_gpu) -> np.ndarray:
    """Split every component ABOVE ``max_transcripts`` via Leiden / threshold cut.

    Components already <= cap are left untouched (never force-split), which is what keeps small
    elongated chains intact instead of being carved into round beads.
    """
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


# --------------------------------------------------------------------------------------
# Contact-interface (chain-only) community merge
# --------------------------------------------------------------------------------------
def _flatten(parent: np.ndarray) -> np.ndarray:
    """Vectorised union-find: collapse every node to its root by pointer jumping."""
    while True:
        nxt = parent[parent]
        if np.array_equal(nxt, parent):
            return parent
        parent = nxt


def _merge_and_finalize(labels, src, dst, emb, config) -> np.ndarray:
    """Contact-interface, mutual-best-neighbour region merge, then drop sub-min communities.

    Replaces the old global mean-embedding (centroid) merge. For each pair of ADJACENT
    communities (sharing >= 1 surviving inter-community mutual-kNN edge) we compute a LOCAL
    contact score: the mean embedding cosine over only the boundary transcripts that form the
    A-B edges (the contact interface). Each round, every (current) community keeps a single
    best admissible neighbour; mutually-best pairs form a matching and are merged. Iterates to a
    fixed point. Non-adjacent pairs can never merge, so two segments of one neurite (drifting
    centroids but an agreeing contact interface) merge end-to-end while two unlike cells touching
    at a thin low-cosine interface stay separate.
    """
    _, labels = np.unique(labels, return_inverse=True)
    K = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=K).astype(np.int64)

    parent = np.arange(K, dtype=np.int64)
    # Inter-community directed edges (both directions present from symmetrisation).
    cs, cd = labels[src], labels[dst]
    inter = cs != cd
    e_src = src[inter]
    e_dst = dst[inter]

    if e_src.size:
        for _ in range(64):  # safety bound; merges roughly halve live roots per round
            root = _flatten(parent)
            ra = root[labels[e_src]]
            rb = root[labels[e_dst]]
            live = ra != rb
            if not live.any():
                break
            es_l, ra_l, rb_l = e_src[live], ra[live], rb[live]

            # Canonical (low, high) root-pair key per inter-root edge.
            lo = np.minimum(ra_l, rb_l)
            hi = np.maximum(ra_l, rb_l)
            pair_key = lo * np.int64(K) + hi
            uk, inv = np.unique(pair_key, return_inverse=True)
            pa, pb = uk // K, uk % K

            # Contact score per root-pair: mean embedding of the boundary tx on this interface,
            # then cosine between the two sides' contact means.
            P = uk.size
            D = emb.shape[1]
            # Sum endpoint embeddings on each side of every interface, weighted by edge count.
            sum_a = np.zeros((P, D), dtype=np.float64)
            sum_b = np.zeros((P, D), dtype=np.float64)
            cnt_a = np.zeros(P, dtype=np.float64)
            cnt_b = np.zeros(P, dtype=np.float64)
            # endpoint on the `lo` side vs the `hi` side of each edge
            on_lo = ra_l == lo  # True if e_src endpoint is the `lo` root
            src_emb = emb[es_l]
            dst_emb = emb[e_dst[live]]
            # lo-side contact tx
            np.add.at(sum_a, inv[on_lo], src_emb[on_lo])
            np.add.at(cnt_a, inv[on_lo], 1.0)
            np.add.at(sum_a, inv[~on_lo], dst_emb[~on_lo])
            np.add.at(cnt_a, inv[~on_lo], 1.0)
            # hi-side contact tx
            np.add.at(sum_b, inv[on_lo], dst_emb[on_lo])
            np.add.at(cnt_b, inv[on_lo], 1.0)
            np.add.at(sum_b, inv[~on_lo], src_emb[~on_lo])
            np.add.at(cnt_b, inv[~on_lo], 1.0)
            mean_a = sum_a / np.maximum(cnt_a, 1.0)[:, None]
            mean_b = sum_b / np.maximum(cnt_b, 1.0)[:, None]
            mean_a /= np.linalg.norm(mean_a, axis=1, keepdims=True) + 1e-12
            mean_b /= np.linalg.norm(mean_b, axis=1, keepdims=True) + 1e-12
            contact = np.einsum("ij,ij->i", mean_a, mean_b)

            rsize = np.bincount(root, weights=counts, minlength=K)
            ok = (contact >= config.merge_threshold) & (
                rsize[pa] + rsize[pb] <= config.max_transcripts
            )
            if not ok.any():
                break
            pa_ok, pb_ok, sc_ok = pa[ok], pb[ok], contact[ok]

            # Each root's unique best edge: highest contact, ties broken by smallest edge id.
            eid = np.arange(pa_ok.size)
            endp = np.concatenate([pa_ok, pb_ok])
            ecos = np.concatenate([sc_ok, sc_ok])
            eidx = np.concatenate([eid, eid])
            best = np.full(K, -np.inf)
            np.maximum.at(best, endp, ecos)
            at_best = ecos >= best[endp] - 1e-9
            chosen = np.full(K, pa_ok.size, dtype=np.int64)  # sentinel = no edge
            np.minimum.at(chosen, endp[at_best], eidx[at_best])
            mutual = (chosen[pa_ok] == eid) & (chosen[pb_ok] == eid)  # a matching
            if not mutual.any():
                break
            parent[np.maximum(pa_ok[mutual], pb_ok[mutual])] = np.minimum(
                pa_ok[mutual], pb_ok[mutual]
            )

    root = _flatten(parent)
    final = root[labels]
    rsize = np.bincount(final, minlength=K)
    out = np.full(labels.shape[0], -1, dtype=np.int64)
    keep = rsize[final] >= config.min_transcripts
    if keep.any():
        _, out_ids = np.unique(final[keep], return_inverse=True)
        out[keep] = out_ids
    return out


# --------------------------------------------------------------------------------------
# Quickshift backend (DEFAULT) -- embedding-density mode-seeking forest
# --------------------------------------------------------------------------------------
def _knn_with_cos(xy: np.ndarray, emb: np.ndarray, k: int, use_gpu: bool):
    """One spatial k-NN. Returns host ``(idx, dist, cos)`` each ``(n, k)``.

    ``cos[i, j]`` is the embedding cosine between ``i`` and its j-th spatial neighbour
    ``idx[i, j]``. The cosine is computed on the kNN device (GPU when available) so the
    transient ``(n*k, D)`` gather never lands on the host.
    """
    n = xy.shape[0]
    if use_gpu:
        xy_d = cp.asarray(xy, dtype=cp.float32)
        emb_d = cp.asarray(emb, dtype=cp.float32)
        nn = cuml.neighbors.NearestNeighbors(n_neighbors=k)
        nn.fit(xy_d)
        dist_d, idx_d = nn.kneighbors(xy_d)
        rows = cp.repeat(cp.arange(n, dtype=cp.int64), k)
        cols = idx_d.reshape(-1)
        cos_d = cp.einsum("ed,ed->e", emb_d[rows], emb_d[cols]).reshape(n, k)
        return (
            cp.asnumpy(idx_d).astype(np.int64),
            cp.asnumpy(dist_d).astype(np.float64),
            cp.asnumpy(cos_d).astype(np.float64),
        )
    tree = KDTree(xy)
    dist, idx = tree.query(xy, k=k)
    dist = np.atleast_2d(dist).astype(np.float64)
    idx = np.atleast_2d(idx).astype(np.int64)
    rows = np.repeat(np.arange(n, dtype=np.int64), idx.shape[1])
    cols = idx.reshape(-1)
    cos = np.einsum("ed,ed->e", emb[rows], emb[cols]).reshape(idx.shape).astype(np.float64)
    return idx, dist, cos


def _quickshift_cluster(
    xy: np.ndarray, emb: np.ndarray, config: FragmentConfig, use_gpu: bool
) -> np.ndarray:
    """Embedding-density mode-seeking (quickshift) over orphan transcripts.

    The model's learned embedding is sharp at cell boundaries, so we use it twice:

    1. **Density (``rho``)** = local *cellness*: a Gaussian-in-space, embedding-affinity-weighted
       sum over each transcript's spatial neighbourhood. High in coherent cell interiors / along a
       coherent process, low at boundaries and for isolated noise.
    2. **Parenting**: each transcript links to the *nearest higher-density* neighbour in a joint
       metric ``d / (0.5*(cos+1))`` -- but only to neighbours that are spatially within
       ``max_dist`` AND embedding-coherent (``cos >= edge_threshold``). A point with no admissible
       higher-density neighbour is a *mode* (root).

    Each node has exactly one parent, so the result is a forest; its trees are the fragments
    (roots = density modes ~ cell cores). Because admissible links cannot cross a low-cosine seam
    or a spatial gap, two touching unlike cells stay separate while a thin neurite -- whose beads
    each link to the next-denser bead along the chain -- is recovered whole and elongated. One k-NN
    + vectorised density/argmin + pointer-jumping union-find: near-linear, deterministic, no
    resolution and no density tree.
    """
    n, _ = emb.shape
    k = min(config.n_neighbors + 1, n)  # +1: self is included
    idx, dist, cos = _knn_with_cos(xy, emb, k, use_gpu)

    rows = np.arange(n)[:, None]
    self_mask = idx == rows

    # Robust local spatial scale h = median nearest (non-self) neighbour distance.
    nn_dist = np.where(self_mask, np.inf, dist).min(axis=1)
    finite = nn_dist[np.isfinite(nn_dist)]
    h = float(np.median(finite)) if finite.size else 1.0
    if not np.isfinite(h) or h <= 0:
        h = 1.0

    # --- coherent edges -----------------------------------------------------------------
    # An edge is *coherent* iff the neighbour is not self and the embedding cosine clears
    # ``edge_threshold``. Coherence -- not a spatial cap -- is the separator: edges never cross a
    # cell seam (low cosine), so unlike cells fall into disjoint basins even when touching, while a
    # generous spatial reach still lets a basin climb to its true mode. An optional hard spatial cap
    # (``quickshift_max_dist``) is applied only when explicitly set.
    coherent = (~self_mask) & (cos >= config.edge_threshold)
    if config.quickshift_max_dist is not None:
        coherent &= dist <= config.quickshift_max_dist

    # --- density (cellness) -------------------------------------------------------------
    # Gaussian-in-space KDE over coherent neighbours, then a couple of MEAN diffusion (low-pass)
    # steps so the field is smooth enough to admit an uphill direction (raw kNN density is bumpy).
    sigma = max(config.quickshift_max_dist_factor * h, h)
    kern = np.where(coherent, np.exp(-(dist ** 2) / (2.0 * sigma * sigma)), 0.0)
    rho = kern.sum(axis=1)
    deg = coherent.sum(axis=1).astype(np.float64)
    for _ in range(2):
        rho = (rho + np.where(coherent, rho[idx], 0.0).sum(axis=1)) / (1.0 + deg)

    # --- mode-seeking forest (quickshift) -----------------------------------------------
    # Each transcript links to its NEAREST strictly-higher-density coherent neighbour; a point with
    # no higher coherent neighbour is a mode (root). One parent each + strictly increasing density
    # => acyclic forest whose trees are density basins.
    ar = np.arange(n)
    higher = (np.where(coherent, rho[idx], -np.inf)) > rho[:, None]
    nearest = np.where(higher, dist, np.inf)
    best_col = np.argmin(nearest, axis=1)
    has_parent = np.isfinite(nearest[ar, best_col])
    parent = np.where(has_parent, idx[ar, best_col], ar).astype(np.int64)
    basin = _flatten(parent)
    _, basin = np.unique(basin, return_inverse=True)  # dense basin ids 0..B-1
    B = int(basin.max()) + 1 if basin.size else 0

    # --- persistence merge (ToMATo) -----------------------------------------------------
    # A uniform cell is a flat plateau and shatters into many basins; two genuinely distinct cells
    # are separated by a deep density valley. Merge two adjacent basins across their highest
    # connecting coherent saddle unless the shallower basin's *prominence* (peak - saddle) exceeds
    # tau (a fraction of the global peak). Coherence-disjoint basins (touching UNLIKE cells) share
    # no edge and never merge, whatever tau is.
    if B > 1:
        peak = np.full(B, -np.inf)
        np.maximum.at(peak, basin, rho)
        flat = coherent.ravel()
        ii = np.repeat(ar, idx.shape[1])[flat]
        jj = idx.ravel()[flat]
        ba, bb = basin[ii], basin[jj]
        cross = ba != bb
        if cross.any():
            ba, bb = ba[cross], bb[cross]
            sad = np.minimum(rho[ii[cross]], rho[jj[cross]])
            lo = np.minimum(ba, bb)
            hi = np.maximum(ba, bb)
            key = lo * np.int64(B) + hi
            # Highest connecting saddle per basin pair.
            uniq_key, inv = np.unique(key, return_inverse=True)
            pair_sad = np.full(uniq_key.size, -np.inf)
            np.maximum.at(pair_sad, inv, sad)
            pa, pb = uniq_key // np.int64(B), uniq_key % np.int64(B)

            tau = config.quickshift_persistence * float(peak.max())
            bparent = np.arange(B, dtype=np.int64)
            bpeak = peak.copy()

            def _find(x):
                root = x
                while bparent[root] != root:
                    root = bparent[root]
                while bparent[x] != root:
                    bparent[x], x = root, bparent[x]
                return root

            for p in np.argsort(-pair_sad, kind="stable"):  # merge across highest saddles first
                ra, rb = _find(int(pa[p])), _find(int(pb[p]))
                if ra == rb:
                    continue
                if bpeak[ra] < bpeak[rb]:
                    ra, rb = rb, ra  # ra = higher peak
                if bpeak[rb] - pair_sad[p] < tau:  # shallower basin not prominent -> merge
                    bparent[rb] = ra
                    bpeak[ra] = max(bpeak[ra], bpeak[rb])
            basin = np.array([_find(int(b)) for b in basin], dtype=np.int64)

    # --- noise floor + dense relabel ----------------------------------------------------
    sizes = np.bincount(basin, minlength=int(basin.max()) + 1 if basin.size else 1)
    out = np.full(n, -1, dtype=np.int64)
    keep = sizes[basin] >= config.min_transcripts
    if keep.any():
        _, out_ids = np.unique(basin[keep], return_inverse=True)
        out[keep] = out_ids
    return out


# --------------------------------------------------------------------------------------
# HDBSCAN backend (grafted bake-off alternative)
# --------------------------------------------------------------------------------------
def _hdbscan_cluster(xy: np.ndarray, emb: np.ndarray, config: FragmentConfig, use_gpu: bool) -> np.ndarray:
    """Cluster orphans with HDBSCAN on a co-scaled joint space+embedding matrix.

    ``F = [xy / space_scale, normalize(emb) * emb_weight * sqrt(2 / D)]``. The ``sqrt(2 / D)``
    normalisation makes one nuclear radius of space cost ~1 unit and a full embedding flip cost
    ~sqrt(2) units, so neither modality alone bridges or severs a chain. No size-cap split.
    """
    n, D = emb.shape
    emb_unit = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    scale = config.emb_weight * math.sqrt(2.0 / max(D, 1))
    F = np.concatenate([xy / config.space_scale, emb_unit * scale], axis=1).astype(np.float32)

    min_samples = config.min_samples
    if min_samples is None:
        min_samples = max(5, config.min_transcripts // 4)
    min_samples = min(min_samples, n)
    min_cluster_size = max(2, min(config.min_transcripts, n))

    if use_gpu:
        clusterer = cuml.cluster.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=config.cluster_selection,
        )
        labels = clusterer.fit_predict(cp.asarray(F))
        labels = cp.asnumpy(labels).astype(np.int64)
    else:
        from sklearn.cluster import HDBSCAN

        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=config.cluster_selection,
        )
        labels = clusterer.fit_predict(F).astype(np.int64)

    # Enforce the min-size noise floor and relabel to a dense 0..k-1 range; -1 = noise.
    out = np.full(n, -1, dtype=np.int64)
    valid = labels >= 0
    if valid.any():
        sizes = np.bincount(labels[valid])
        keep = valid & (sizes[np.where(valid, labels, 0)] >= config.min_transcripts)
        if keep.any():
            _, out_ids = np.unique(labels[keep], return_inverse=True)
            out[keep] = out_ids
    if config.max_transcripts and valid.any():  # QC/log only: HDBSCAN does no size-cap split
        big = np.bincount(out[out >= 0]) if (out >= 0).any() else np.array([])
        if big.size and big.max() > config.max_transcripts:
            warnings.warn(
                f"HDBSCAN produced a fragment with {int(big.max())} transcripts "
                f"(> max_transcripts={config.max_transcripts}); not split.",
                RuntimeWarning,
                stacklevel=2,
            )
    return out


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------
def assign_fragments(
    xy: np.ndarray,
    emb: np.ndarray,
    config: FragmentConfig | None = None,
) -> np.ndarray:
    """Assign fragment IDs to a set of unassigned (orphan) transcripts.

    Parameters
    ----------
    xy : np.ndarray
        ``(N, 2)`` transcript coordinates.
    emb : np.ndarray
        ``(N, D)`` GNN transcript embeddings (L2-normalised internally).
    config : FragmentConfig, optional
        Hyperparameters; defaults to ``FragmentConfig()`` (``method='leiden'``).

    Returns
    -------
    np.ndarray
        ``(N,)`` fragment IDs (``-1`` where the transcript is not part of a fragment within
        ``[min_transcripts, max_transcripts]`` / is noise).
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

    if config.method == "quickshift":
        return _quickshift_cluster(xy, emb, config, use_gpu)

    if config.method == "hdbscan":
        return _hdbscan_cluster(xy, emb, config, use_gpu)

    # method == 'leiden'
    src, dst, weight, dist, labels = _build_graph(xy, emb, config, use_gpu)
    if src.size == 0:
        return out

    labels = _enforce_max(labels, src, dst, weight, dist, config, use_gpu)
    return _merge_and_finalize(labels, src, dst, emb, config)
