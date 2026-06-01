"""Tests for ``segger.prediction.fragment`` (CPU fallback path)."""

import numpy as np

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


def _three_blobs(rng, sizes=(80, 120, 200), radius=4.0):
    centers = [(0.0, 0.0), (50.0, 0.0), (0.0, 50.0)]
    xys, embs = [], []
    for c, n in zip(centers, sizes):
        xy, emb = _make_blob(c, n, radius, rng)
        xys.append(xy)
        embs.append(emb)
    return np.vstack(xys), np.vstack(embs), list(sizes)


def _cfg(**kw):
    base = dict(min_transcripts=50, n_neighbors=15, use_gpu=False)
    base.update(kw)
    return FragmentConfig(**base)


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
    # Two spatially-overlapping blobs with opposite embeddings: embedding
    # pruning keeps them as two separate fragments (no spatial fusion).
    rng = np.random.default_rng(5)
    e_pos = np.zeros(64, dtype=np.float32); e_pos[0] = 1.0
    e_neg = -e_pos
    xy_a, emb_a = _make_blob((0.0, 0.0), 150, radius=3.0, rng=rng, mean_emb=e_pos)
    xy_b, emb_b = _make_blob((0.0, 0.0), 150, radius=3.0, rng=rng, mean_emb=e_neg)
    xy = np.vstack([xy_a, xy_b])
    emb = np.vstack([emb_a, emb_b])
    labels = assign_fragments(xy, emb, _cfg())
    assert len(np.unique(labels[labels >= 0])) == 2
    # the two fragments correspond to the two embedding groups
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
    # A thin filament of one embedding type must stay one elongated fragment,
    # never be carved into round beads.
    rng = np.random.default_rng(7)
    n = 400
    x = np.linspace(0.0, 120.0, n)
    y = rng.normal(scale=0.3, size=n)
    xy = np.column_stack([x, y]).astype(np.float32)
    e = rng.normal(size=64).astype(np.float32); e /= np.linalg.norm(e)
    emb = (e[None] + rng.normal(scale=0.02, size=(n, 64))).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
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
