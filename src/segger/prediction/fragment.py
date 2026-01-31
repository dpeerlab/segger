"""Fragment mode for grouping unassigned transcripts.

This module implements fragment-based segmentation for transcripts that
were not assigned to any cell during primary segmentation. It uses
transcript-transcript edge similarity and connected components to
create "fragment cells" from spatially proximal unassigned transcripts.
"""

from typing import Optional
import numpy as np
import polars as pl

# Try to import GPU-accelerated connected components
try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
    from cupyx.scipy.sparse.csgraph import connected_components as cc_gpu
    HAS_RAPIDS = True
except ImportError:
    HAS_RAPIDS = False

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as cc_cpu


def compute_fragment_components(
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    similarities: np.ndarray,
    similarity_threshold: float = 0.5,
    use_gpu: bool = True,
) -> dict[int, int]:
    """Compute connected components from transcript-transcript edges.

    Parameters
    ----------
    source_ids : np.ndarray
        Source transcript indices.
    target_ids : np.ndarray
        Target transcript indices.
    similarities : np.ndarray
        Similarity scores for each edge.
    similarity_threshold : float, optional
        Minimum similarity to include edge (default: 0.5).
    use_gpu : bool, optional
        Whether to use GPU acceleration if available (default: True).

    Returns
    -------
    dict[int, int]
        Mapping from transcript index to component label.
    """
    # Filter edges by similarity threshold
    mask = similarities >= similarity_threshold
    src = source_ids[mask]
    dst = target_ids[mask]

    if len(src) == 0:
        return {}

    # Get unique node IDs and create mapping
    unique_ids = np.unique(np.concatenate([src, dst]))
    id_to_idx = {id_: idx for idx, id_ in enumerate(unique_ids)}
    n_nodes = len(unique_ids)

    # Map edges to contiguous indices
    src_idx = np.array([id_to_idx[s] for s in src])
    dst_idx = np.array([id_to_idx[d] for d in dst])

    # Create symmetric adjacency matrix
    data = np.ones(len(src_idx) * 2)
    rows = np.concatenate([src_idx, dst_idx])
    cols = np.concatenate([dst_idx, src_idx])

    # Use GPU or CPU connected components
    if use_gpu and HAS_RAPIDS:
        adj_matrix = cp_csr_matrix(
            (cp.asarray(data), (cp.asarray(rows), cp.asarray(cols))),
            shape=(n_nodes, n_nodes),
        )
        n_components, labels = cc_gpu(adj_matrix, directed=False)
        labels = cp.asnumpy(labels)
    else:
        adj_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(n_nodes, n_nodes),
        )
        n_components, labels = cc_cpu(adj_matrix, directed=False)

    # Map back to original IDs
    return {unique_ids[idx]: int(labels[idx]) for idx in range(n_nodes)}


def apply_fragment_mode(
    segmentation_df: pl.DataFrame,
    tx_tx_edges: pl.DataFrame,
    min_transcripts: int = 5,
    similarity_threshold: float = 0.5,
    use_gpu: bool = True,
    cell_id_column: str = "segger_cell_id",
    transcript_id_column: str = "transcript_id",
    similarity_column: str = "similarity",
) -> pl.DataFrame:
    """Apply fragment mode to group unassigned transcripts into fragment cells.

    Parameters
    ----------
    segmentation_df : pl.DataFrame
        Segmentation results with cell assignments.
    tx_tx_edges : pl.DataFrame
        Transcript-transcript edges with columns for source, target, and similarity.
    min_transcripts : int, optional
        Minimum transcripts per fragment cell (default: 5).
    similarity_threshold : float, optional
        Minimum similarity for tx-tx edges (default: 0.5).
    use_gpu : bool, optional
        Whether to use GPU acceleration (default: True).
    cell_id_column : str, optional
        Column name for cell IDs (default: "segger_cell_id").
    transcript_id_column : str, optional
        Column name for transcript IDs (default: "transcript_id").
    similarity_column : str, optional
        Column name for similarity scores (default: "similarity").

    Returns
    -------
    pl.DataFrame
        Updated segmentation with fragment cell assignments.
    """
    # Find unassigned transcripts
    unassigned_mask = segmentation_df[cell_id_column].is_null()
    unassigned_ids = set(
        segmentation_df
        .filter(unassigned_mask)
        .select(transcript_id_column)
        .to_series()
        .to_list()
    )

    if len(unassigned_ids) == 0:
        return segmentation_df

    # Filter tx-tx edges to only include unassigned transcripts on both ends
    filtered_edges = (
        tx_tx_edges
        .filter(
            pl.col("source").is_in(unassigned_ids) &
            pl.col("target").is_in(unassigned_ids)
        )
    )

    if filtered_edges.height == 0:
        return segmentation_df

    # Extract edge data
    source_ids = filtered_edges.select("source").to_numpy().flatten()
    target_ids = filtered_edges.select("target").to_numpy().flatten()
    similarities = filtered_edges.select(similarity_column).to_numpy().flatten()

    # Compute connected components
    component_labels = compute_fragment_components(
        source_ids=source_ids,
        target_ids=target_ids,
        similarities=similarities,
        similarity_threshold=similarity_threshold,
        use_gpu=use_gpu,
    )

    if not component_labels:
        return segmentation_df

    # Count transcripts per component
    from collections import Counter
    component_counts = Counter(component_labels.values())

    # Filter to components with minimum transcripts
    valid_components = {
        comp for comp, count in component_counts.items()
        if count >= min_transcripts
    }

    if not valid_components:
        return segmentation_df

    # Create fragment cell IDs (prefix with "fragment-" to distinguish)
    # Find max existing cell ID to create non-overlapping fragment IDs
    existing_ids = (
        segmentation_df
        .filter(~unassigned_mask)
        .select(cell_id_column)
        .unique()
        .to_series()
        .to_list()
    )

    # Create mapping from component to fragment cell ID
    fragment_id_map = {}
    for comp in sorted(valid_components):
        fragment_id_map[comp] = f"fragment-{comp}"

    # Create update DataFrame
    updates = []
    for tx_id, comp in component_labels.items():
        if comp in valid_components:
            updates.append({
                transcript_id_column: tx_id,
                f"{cell_id_column}_fragment": fragment_id_map[comp],
            })

    if not updates:
        return segmentation_df

    update_df = pl.DataFrame(updates)

    # Join updates back to segmentation
    result = (
        segmentation_df
        .join(update_df, on=transcript_id_column, how="left")
        .with_columns(
            pl.coalesce([
                pl.col(cell_id_column),
                pl.col(f"{cell_id_column}_fragment"),
            ]).alias(cell_id_column)
        )
        .drop(f"{cell_id_column}_fragment")
    )

    return result
