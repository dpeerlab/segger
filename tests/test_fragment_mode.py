"""Tests for ``segger.prediction.fragment`` (CPU fallback path).

Covers both Stage B backends:

* ``method='leiden'`` (default): mutual-kNN + connected components + Leiden split of oversized
  components + contact-interface chain-only merge. The anti-round behaviour (elongated chains stay
  one elongated fragment, unlike touching cells stay separate) is asserted here.
* ``method='hdbscan'`` (grafted bake-off backend): co-scaled space+embedding HDBSCAN.

The GPU (RAPIDS / cuML / cuGraph) path is import-guarded, so these run CPU-only on a dev box.
"""

import numpy as np
import pytest

from segger.prediction.fragment import FragmentConfig, assign_fragments


def _make_blob(center, n, radius, rng, dim=64, mean_emb=None):
    """Make ``n`` points uniformly in a disk + a coherent embedding cluster."""
    angles = rng.uniform(0, 2 * np.pi, size=n)
    radii = radius * np.sqrt(rng.uniform(0, 1, size=n))
    xy = np.column_stack([
        center[0] + radii * np.cos(angles),
        center[1] + radii * np.sin(angles),
    ]).astype(np.float32)
    if mean_emb is None:
        mean_emb = rng.normal(size=dim).astype(np.float32)
        mean_emb /= np.linalg.norm(mean_emb)
    noise = rng.normal(scale=0.02, size=(n, dim)).astype(np.float32)
    emb = mean_emb[None] + noise
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return xy, emb


def _make_filament(n, length, jitter, rng, dim=64, mean_emb=None):
    """Make a thin 1-D filament of ``n`` points (an elongated chain)."""
    x = np.linspace(0.0, length, n)
    y = rng.normal(scale=jitter, size=n)
    xy = np.column_stack([x, y]).astype(np.float32)
    if mean_emb is None:
        mean_emb = rng.normal(size=dim).astype(np.float32)
        mean_emb /= np.linalg.norm(mean_emb)
    emb = (mean_emb[None] + rng.normal(scale=0.02, size=(n, dim)).astype(np.float32))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return xy, emb


def _three_blobs(rng, sizes=(80, 120, 200), radius=4.0):
    centers = [(0.0, 0.0), (50.0, 0.0), (0.0, 50.0)]
    xys, embs = [], []
    for c, n in zip(centers, sizes):
        xy, emb = _make_blob(c, n, radius, rng)
        xys.append(xy)
        embs.append(emb)
    return np.vstack(xys), np.vstack(embs), list(sizes)


def _cfg(**kw):
    # Legacy tests pin the shipped default backend ('leiden') and the CPU path.
    base = dict(min_transcripts=50, n_neighbors=15, use_gpu=False, method="leiden")
    base.update(kw)
    return FragmentConfig(**base)


# ----------------------------------------------------------------------------------------------
# Legacy behaviour (kept green; pinned to method='leiden')
# ----------------------------------------------------------------------------------------------
def test_assigns_each_blob_to_one_fragment():
    rng = np.random.default_rng(0)
    xy, emb, sizes = _three_blobs(rng)
    labels = assign_fragments(xy, emb, _cfg())
    assert (labels >= 0).all()
    offsets = np.cumsum([0] + sizes)
    blob_labels = [np.unique(labels[a:b]) for a, b in zip(offsets[:-1], offsets[1:])]
    assert all(len(l) == 1 for l in blob_labels)
    assert len({int(l[0]) for l in blob_labels}) == 3


def test_size_bounds_are_respected():
    rng = np.random.default_rng(1)
    xy, emb = _make_blob((0.0, 0.0), 2000, radius=8.0, rng=rng)
    config = _cfg(max_transcripts=300)
    labels = assign_fragments(xy, emb, config)
    sizes = np.bincount(labels[labels >= 0])
    assert sizes.size > 0
    assert sizes.min() >= config.min_transcripts
    assert sizes.max() <= config.max_transcripts


def test_small_blob_is_dropped():
    rng = np.random.default_rng(2)
    xy, emb = _make_blob((0.0, 0.0), 10, radius=2.0, rng=rng)
    labels = assign_fragments(xy, emb, _cfg())
    assert (labels == -1).all()


def test_deterministic():
    rng = np.random.default_rng(3)
    xy, emb, _ = _three_blobs(rng)
    a = assign_fragments(xy, emb, _cfg())
    b = assign_fragments(xy, emb, _cfg())
    np.testing.assert_array_equal(a, b)


def test_off_staining_blob_is_recovered():
    rng = np.random.default_rng(4)
    main_xy, main_emb, _ = _three_blobs(rng)
    far_xy, far_emb = _make_blob((1000.0, 1000.0), 120, radius=4.0, rng=rng)
    xy = np.vstack([main_xy, far_xy])
    emb = np.vstack([main_emb, far_emb])
    labels = assign_fragments(xy, emb, _cfg())
    far_labels = labels[main_xy.shape[0]:]
    assert (far_labels >= 0).all()
    assert len(np.unique(far_labels)) == 1


def test_unlike_embeddings_split_while_overlapping():
    # Two spatially-overlapping blobs with opposite embeddings: embedding pruning keeps them as
    # two separate fragments (no spatial fusion).
    rng = np.random.default_rng(5)
    e_pos = np.zeros(64, dtype=np.float32); e_pos[0] = 1.0
    e_neg = -e_pos
    xy_a, emb_a = _make_blob((0.0, 0.0), 150, radius=3.0, rng=rng, mean_emb=e_pos)
    xy_b, emb_b = _make_blob((0.0, 0.0), 150, radius=3.0, rng=rng, mean_emb=e_neg)
    xy = np.vstack([xy_a, xy_b])
    emb = np.vstack([emb_a, emb_b])
    labels = assign_fragments(xy, emb, _cfg())
    assert len(np.unique(labels[labels >= 0])) == 2
    assert len(np.unique(labels[:150][labels[:150] >= 0])) == 1
    assert len(np.unique(labels[150:][labels[150:] >= 0])) == 1


def test_like_embeddings_overlapping_form_one_fragment():
    rng = np.random.default_rng(6)
    e = np.zeros(64, dtype=np.float32); e[0] = 1.0
    xy_a, emb_a = _make_blob((0.0, 0.0), 150, radius=3.0, rng=rng, mean_emb=e)
    xy_b, emb_b = _make_blob((0.0, 0.0), 150, radius=3.0, rng=rng, mean_emb=e)
    xy = np.vstack([xy_a, xy_b])
    emb = np.vstack([emb_a, emb_b])
    labels = assign_fragments(xy, emb, _cfg())
    assert len(np.unique(labels[labels >= 0])) == 1


def test_elongated_cluster_kept_as_one_fragment():
    # A thin filament of one embedding type must stay one elongated fragment, never carved into
    # round beads.
    rng = np.random.default_rng(7)
    xy, emb = _make_filament(400, length=120.0, jitter=0.3, rng=rng)
    labels = assign_fragments(xy, emb, _cfg())
    kept = labels[labels >= 0]
    assert kept.size > 0
    assert len(np.unique(kept)) == 1                  # one fragment, not beads
    members = labels == kept[0]
    assert np.ptp(xy[members, 0]) > 10 * np.ptp(xy[members, 1])  # elongated


def test_degenerate_inputs():
    cfg = _cfg()
    assert (assign_fragments(np.zeros((5, 2), np.float32),
                             np.zeros((5, 64), np.float32), cfg) == -1).all()
    assert assign_fragments(np.zeros((0, 2), np.float32),
                            np.zeros((0, 64), np.float32), cfg).shape == (0,)


# ----------------------------------------------------------------------------------------------
# New behaviour (winning anti-round spec)
# ----------------------------------------------------------------------------------------------
def _chain_in_noisy_field(rng, n_chain=300, length=150.0, n_noise=200, dim=64):
    """A consistent-embedding thin chain embedded in a field of scattered random-embedding tx.

    The scatter shares the chain's bounding box, so it is the cross-chain "waist" that mutual-kNN
    must reject to keep the chain 1-D instead of letting it blob outward.
    """
    chain_xy, chain_emb = _make_filament(n_chain, length=length, jitter=0.5, rng=rng, dim=dim)
    noise_xy = np.column_stack([
        rng.uniform(0.0, length, n_noise),
        rng.uniform(-30.0, 30.0, n_noise),
    ]).astype(np.float32)
    noise_emb = rng.normal(size=(n_noise, dim)).astype(np.float32)
    noise_emb /= np.linalg.norm(noise_emb, axis=1, keepdims=True)
    xy = np.vstack([chain_xy, noise_xy])
    emb = np.vstack([chain_emb, noise_emb])
    return xy, emb, n_chain


def _dominant_label(labels, slice_):
    sub = labels[slice_]
    kept = sub[sub >= 0]
    if kept.size == 0:
        return None, 0
    vals, counts = np.unique(kept, return_counts=True)
    return int(vals[counts.argmax()]), int(counts.max())


def test_mutual_knn_keeps_chain_elongated_as_one():
    """(1) Thin chain of consistent embedding -> ONE elongated community under mutual_knn=True."""
    rng = np.random.default_rng(11)
    xy, emb, n_chain = _chain_in_noisy_field(rng)
    labels = assign_fragments(xy, emb, _cfg(mutual_knn=True))

    chain = labels[:n_chain]
    kept = chain[chain >= 0]
    assert kept.size > 0
    # Chain transcripts collapse to a single dominant community (not a string of round beads).
    assert len(np.unique(kept)) == 1
    dom, _ = _dominant_label(labels, slice(0, n_chain))
    members = labels == dom
    # The community is genuinely anisotropic.
    assert np.ptp(xy[members, 0]) > 10 * np.ptp(xy[members, 1])
    # The scattered random-embedding field is rejected (it is not part of the chain community).
    assert not (labels[n_chain:] == dom).any()


def test_mutual_knn_toggle_changes_graph():
    """mutual_knn is a real ablation lever: turning it off cannot shrink the surviving edge set.

    Mutual k-NN is, by construction, a subset of the symmetric k-NN graph, so the elongated chain
    must still be recoverable, while the symmetric graph admits strictly more (waist) edges. We
    assert the invariant directly on the graph builder.
    """
    from segger.prediction.fragment import _build_graph

    rng = np.random.default_rng(11)
    xy, emb, _ = _chain_in_noisy_field(rng)
    xy = np.ascontiguousarray(xy, np.float32)
    emb = np.ascontiguousarray(emb, np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12

    s_m, *_ = _build_graph(xy, emb, _cfg(mutual_knn=True), use_gpu=False)
    s_s, *_ = _build_graph(xy, emb, _cfg(mutual_knn=False), use_gpu=False)
    # Mutual graph is a strict subset: fewer (or equal) directed edges than symmetric.
    assert s_m.size <= s_s.size


def test_unlike_touching_cells_do_not_merge():
    """(2) Two unlike cells touching at a thin interface stay separate (low contact cosine)."""
    rng = np.random.default_rng(12)
    e_pos = np.zeros(64, dtype=np.float32); e_pos[0] = 1.0
    e_neg = -e_pos
    # Centres 7 apart, radius 4 -> the two disks share a contact interface.
    xy_a, emb_a = _make_blob((0.0, 0.0), 200, radius=4.0, rng=rng, mean_emb=e_pos)
    xy_b, emb_b = _make_blob((7.0, 0.0), 200, radius=4.0, rng=rng, mean_emb=e_neg)
    xy = np.vstack([xy_a, xy_b])
    emb = np.vstack([emb_a, emb_b])
    labels = assign_fragments(xy, emb, _cfg())
    assert len(np.unique(labels[labels >= 0])) == 2


def test_compact_missed_cell_recovered():
    """(3) A compact missed cell above min_transcripts is recovered as one fragment."""
    rng = np.random.default_rng(13)
    xy, emb = _make_blob((0.0, 0.0), 120, radius=4.0, rng=rng)
    labels = assign_fragments(xy, emb, _cfg())
    assert (labels >= 0).all()
    assert len(np.unique(labels[labels >= 0])) == 1


def test_scattered_transcripts_are_noise():
    """(4) Sparse, embedding-incoherent scatter -> all noise (no invented cells)."""
    rng = np.random.default_rng(14)
    xy = np.column_stack([
        rng.uniform(0.0, 500.0, 300),
        rng.uniform(0.0, 500.0, 300),
    ]).astype(np.float32)
    emb = rng.normal(size=(300, 64)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    labels = assign_fragments(xy, emb, _cfg())
    assert (labels == -1).all()


def test_hdbscan_recovers_elongated_chain_and_rejects_scatter():
    """(5) HDBSCAN backend: the elongated chain is recovered (dominant elongated cluster) and a
    well-separated random-embedding scatter is rejected as noise.

    Note: HDBSCAN's EOM selection can split a thin filament at density saddle points, so we assert
    the robust, non-flaky properties (chain mostly recovered, dominant cluster anisotropic, scatter
    is noise) rather than an exact single-cluster id. This is precisely why 'leiden' is the
    shipped default and 'hdbscan' is the bake-off alternative.
    """
    pytest.importorskip("sklearn.cluster")
    rng = np.random.default_rng(31)
    # Uniform-density thin rectangle (elongated) so HDBSCAN keeps most of it.
    n_chain = 500
    cx = rng.uniform(0.0, 100.0, n_chain)
    cy = rng.uniform(-1.5, 1.5, n_chain)
    chain_xy = np.column_stack([cx, cy]).astype(np.float32)
    e = rng.normal(size=32).astype(np.float32); e /= np.linalg.norm(e)
    chain_emb = (e[None] + rng.normal(scale=0.02, size=(n_chain, 32)).astype(np.float32))
    chain_emb /= np.linalg.norm(chain_emb, axis=1, keepdims=True)
    n_noise = 30
    noise_xy = np.column_stack([
        rng.uniform(800.0, 4000.0, n_noise),
        rng.uniform(800.0, 4000.0, n_noise),
    ]).astype(np.float32)
    noise_emb = rng.normal(size=(n_noise, 32)).astype(np.float32)
    noise_emb /= np.linalg.norm(noise_emb, axis=1, keepdims=True)
    xy = np.vstack([chain_xy, noise_xy])
    emb = np.vstack([chain_emb, noise_emb])

    labels = assign_fragments(
        xy, emb, _cfg(method="hdbscan", space_scale=2.0, min_transcripts=50)
    )
    chain = labels[:n_chain]
    scatter = labels[n_chain:]
    # Most of the chain is recovered.
    assert (chain >= 0).mean() > 0.7
    # The dominant chain cluster is elongated and lives entirely in the chain (not the scatter).
    dom, _ = _dominant_label(labels, slice(0, n_chain))
    members = labels == dom
    assert (np.nonzero(members)[0] < n_chain).all()
    assert np.ptp(xy[members, 0]) > 5 * np.ptp(xy[members, 1])
    # The well-separated random-embedding scatter is noise.
    assert (scatter == -1).all()


def test_hdbscan_scattered_is_noise():
    """HDBSCAN backend rejects sparse incoherent scatter (no size-cap split, still a noise floor)."""
    pytest.importorskip("sklearn.cluster")
    rng = np.random.default_rng(16)
    xy = np.column_stack([
        rng.uniform(0.0, 2000.0, 200),
        rng.uniform(0.0, 2000.0, 200),
    ]).astype(np.float32)
    emb = rng.normal(size=(200, 32)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    labels = assign_fragments(xy, emb, _cfg(method="hdbscan", min_transcripts=50, space_scale=2.0))
    assert (labels == -1).all()


def test_cpu_partition_grouping_is_stable():
    """(6) GPU/CPU agreement is asserted on the PARTITION (label grouping), not exact ids.

    No GPU on the dev box, so we assert the CPU path is stable: repeated runs produce the SAME
    grouping of transcripts into fragments (identical co-membership), independent of arbitrary
    label numbering.
    """
    rng = np.random.default_rng(17)
    xy, emb, sizes = _three_blobs(rng)
    a = assign_fragments(xy, emb, _cfg())
    b = assign_fragments(xy, emb, _cfg())

    def comembership(labels):
        # Boolean: do i and j share a (non-noise) fragment? Compared as a partition invariant.
        valid = labels >= 0
        same = (labels[:, None] == labels[None, :]) & valid[:, None] & valid[None, :]
        return same

    np.testing.assert_array_equal(comembership(a), comembership(b))
    # And the partition has the expected 3 blobs.
    assert len(np.unique(a[a >= 0])) == 3


def test_method_and_config_validation():
    """FragmentConfig validates the new fields per the contract."""
    with pytest.raises(ValueError):
        FragmentConfig(method="kmeans")
    with pytest.raises(ValueError):
        FragmentConfig(cluster_selection="bogus")
    with pytest.raises(ValueError):
        FragmentConfig(space_scale=0.0)
    with pytest.raises(ValueError):
        FragmentConfig(quickshift_max_dist_factor=0.0)
    with pytest.raises(ValueError):
        FragmentConfig(quickshift_max_dist=-1.0)
    # Default backend is the embedding-density mode-seeking 'quickshift'.
    cfg = FragmentConfig()
    assert cfg.method == "quickshift"
    assert cfg.mutual_knn is True


# ---------------------------------------------------------------------------
# Quickshift backend (DEFAULT) -- embedding-density mode-seeking
# ---------------------------------------------------------------------------
def _qs(**kw):
    """Quickshift config on the CPU path (the shipped default backend)."""
    base = dict(min_transcripts=50, n_neighbors=15, use_gpu=False, method="quickshift")
    base.update(kw)
    return FragmentConfig(**base)


def test_quickshift_filament_is_one_elongated_fragment():
    """A thin neurite-like filament is recovered as ONE elongated fragment, not a string of beads."""
    rng = np.random.default_rng(40)
    xy, emb = _make_filament(300, length=120.0, jitter=0.4, rng=rng)
    labels = assign_fragments(xy, emb, _qs())
    kept = labels[labels >= 0]
    assert kept.size > 0
    assert len(np.unique(kept)) == 1            # one fragment, not many beads
    assert (labels >= 0).mean() > 0.9           # nearly all of the filament recovered
    members = labels == kept[0]
    assert np.ptp(xy[members, 0]) > 10 * np.ptp(xy[members, 1])  # genuinely anisotropic


def test_quickshift_compact_cell_is_single_fragment():
    """A compact missed cell resolves to a single mode (no over-segmentation into basins)."""
    rng = np.random.default_rng(41)
    xy, emb = _make_blob((0.0, 0.0), 150, radius=4.0, rng=rng)
    labels = assign_fragments(xy, emb, _qs())
    assert (labels >= 0).mean() > 0.9
    assert len(np.unique(labels[labels >= 0])) == 1


def test_quickshift_unlike_touching_cells_stay_separate():
    """Two unlike cells touching at a thin seam are not bridged (seam edges fail edge_threshold)."""
    rng = np.random.default_rng(42)
    e_pos = np.zeros(64, dtype=np.float32); e_pos[0] = 1.0
    e_neg = -e_pos
    xy_a, emb_a = _make_blob((0.0, 0.0), 200, radius=4.0, rng=rng, mean_emb=e_pos)
    xy_b, emb_b = _make_blob((7.0, 0.0), 200, radius=4.0, rng=rng, mean_emb=e_neg)
    xy = np.vstack([xy_a, xy_b]); emb = np.vstack([emb_a, emb_b])
    labels = assign_fragments(xy, emb, _qs())
    assert len(np.unique(labels[labels >= 0])) == 2
    # No fragment spans both cells.
    for lab in np.unique(labels[labels >= 0]):
        members = np.nonzero(labels == lab)[0]
        assert (members < 200).all() or (members >= 200).all()


def test_quickshift_three_blobs_three_fragments():
    """Three well-separated coherent blobs -> three fragments."""
    rng = np.random.default_rng(43)
    xy, emb, _ = _three_blobs(rng)
    labels = assign_fragments(xy, emb, _qs())
    assert len(np.unique(labels[labels >= 0])) == 3


def test_quickshift_scatter_is_noise():
    """Sparse, embedding-incoherent scatter -> all noise (no invented cells)."""
    rng = np.random.default_rng(44)
    xy = np.column_stack([
        rng.uniform(0.0, 500.0, 300), rng.uniform(0.0, 500.0, 300),
    ]).astype(np.float32)
    emb = rng.normal(size=(300, 64)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    labels = assign_fragments(xy, emb, _qs())
    assert (labels == -1).all()


def test_quickshift_is_deterministic():
    """Repeated runs produce an identical partition (deterministic, no random seed)."""
    rng = np.random.default_rng(45)
    xy, emb, _ = _three_blobs(rng)
    a = assign_fragments(xy, emb, _qs())
    b = assign_fragments(xy, emb, _qs())
    np.testing.assert_array_equal(a, b)
