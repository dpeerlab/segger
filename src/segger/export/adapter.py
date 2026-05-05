"""Adapter to convert model predictions to export-compatible format.

This module bridges the gap between LitISTEncoder.predict_step() output
and the seg2explorer functions for Xenium Explorer export.
"""

from typing import Optional, Union
import pandas as pd
import polars as pl
import torch


def predictions_to_dataframe(
    src_idx: torch.Tensor,
    seg_idx: torch.Tensor,
    max_sim: torch.Tensor,
    gen_idx: torch.Tensor,
    transcript_data: Union[pd.DataFrame, pl.DataFrame],
    min_similarity: float = 0.5,
    x_column: str = "x",
    y_column: str = "y",
    gene_column: str = "feature_name",
) -> pd.DataFrame:
    """Convert prediction tensors to seg2explorer-compatible DataFrame.

    This function takes the output from LitISTEncoder.predict_step() and
    combines it with the original transcript data to create a DataFrame
    suitable for Xenium Explorer export.

    Parameters
    ----------
    src_idx : torch.Tensor
        Transcript indices from prediction, shape (N,).
    seg_idx : torch.Tensor
        Assigned boundary/cell indices, shape (N,). Value of -1 indicates
        unassigned transcripts.
    max_sim : torch.Tensor
        Maximum similarity scores, shape (N,).
    gen_idx : torch.Tensor
        Gene indices for each transcript, shape (N,).
    transcript_data : Union[pd.DataFrame, pl.DataFrame]
        Original transcript DataFrame with coordinates.
    min_similarity : float
        Minimum similarity threshold for valid assignments.
    x_column : str
        Column name for x coordinates.
    y_column : str
        Column name for y coordinates.
    gene_column : str
        Column name for gene/feature names.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - row_index: Original transcript index
        - x: X coordinate
        - y: Y coordinate
        - seg_cell_id: Assigned cell ID (or -1 if unassigned)
        - similarity: Assignment confidence score
        - feature_name: Gene name
    """
    # Convert to numpy
    src_idx_np = src_idx.cpu().numpy()
    seg_idx_np = seg_idx.cpu().numpy()
    max_sim_np = max_sim.cpu().numpy()

    # Filter by similarity threshold
    valid_mask = (seg_idx_np >= 0) & (max_sim_np >= min_similarity)

    # Convert Polars to pandas if needed
    if isinstance(transcript_data, pl.DataFrame):
        transcript_data = transcript_data.to_pandas()

    # Build result DataFrame
    result = pd.DataFrame({
        "row_index": src_idx_np,
        "seg_cell_id": seg_idx_np,
        "similarity": max_sim_np,
    })

    # Mark low-similarity assignments as unassigned
    result.loc[~valid_mask, "seg_cell_id"] = -1

    # Merge with original transcript data for coordinates
    if "row_index" in transcript_data.columns:
        # Use existing row_index
        result = result.merge(
            transcript_data[["row_index", x_column, y_column, gene_column]],
            on="row_index",
            how="left",
        )
    else:
        # Use index as row_index
        transcript_data = transcript_data.reset_index()
        transcript_data = transcript_data.rename(columns={"index": "row_index"})
        result = result.merge(
            transcript_data[["row_index", x_column, y_column, gene_column]],
            on="row_index",
            how="left",
        )

    # Rename columns for consistency
    result = result.rename(columns={
        gene_column: "feature_name",
        x_column: "x",
        y_column: "y",
    })

    return result


def collect_predictions(
    predictions: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect predictions from multiple batches.

    Parameters
    ----------
    predictions : list[tuple]
        List of (src_idx, seg_idx, max_sim, gen_idx) tuples from predict_step.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Concatenated (src_idx, seg_idx, max_sim, gen_idx) tensors.
    """
    src_indices = []
    seg_indices = []
    similarities = []
    gene_indices = []

    for src_idx, seg_idx, max_sim, gen_idx in predictions:
        src_indices.append(src_idx)
        seg_indices.append(seg_idx)
        similarities.append(max_sim)
        gene_indices.append(gen_idx)

    return (
        torch.cat(src_indices),
        torch.cat(seg_indices),
        torch.cat(similarities),
        torch.cat(gene_indices),
    )


def filter_assigned_transcripts(
    seg_df: pd.DataFrame,
    cell_id_column: str = "seg_cell_id",
) -> pd.DataFrame:
    """Filter DataFrame to only include assigned transcripts.

    Parameters
    ----------
    seg_df : pd.DataFrame
        Segmentation result DataFrame.
    cell_id_column : str
        Column name for cell IDs.

    Returns
    -------
    pd.DataFrame
        DataFrame with only assigned transcripts.
    """
    return seg_df[seg_df[cell_id_column] >= 0].copy()
