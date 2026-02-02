"""Xenium Explorer export functionality.

This module converts segmentation results into Xenium Explorer-compatible
Zarr format for visualization and validation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json

import numpy as np
import pandas as pd
import polars as pl
import zarr
from pqdm.processes import pqdm
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from tqdm import tqdm
from zarr.storage import ZipStore

from .boundary import generate_boundary


def get_flatten_version(
    polygon_vertices: List[List[Tuple[float, float]]],
    max_value: int = 21,
) -> np.ndarray:
    """Standardize list of polygon vertices to a fixed shape.

    Parameters
    ----------
    polygon_vertices : List[List[Tuple[float, float]]]
        List of polygon coordinate lists.
    max_value : int
        Max number of coordinates per polygon.

    Returns
    -------
    np.ndarray
        Padded or truncated array of polygon vertices, shape (N, max_value, 2).
    """
    flattened = []
    for vertices in polygon_vertices:
        if len(vertices) < 3:
            continue
        if isinstance(vertices, np.ndarray):
            vertices = vertices.tolist()

        if len(vertices) > max_value:
            flattened.append(vertices[:max_value])
        else:
            flattened.append(vertices + [vertices[0]] * (max_value - len(vertices)))
    return np.array(flattened, dtype=np.float32)


def get_indices_indptr(input_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get sparse matrix representation for cluster assignments.

    Parameters
    ----------
    input_array : np.ndarray
        Array of cluster labels.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Indices and indptr arrays for CSR-like representation.
    """
    clusters = sorted(np.unique(input_array[input_array != 0]))
    indptr = np.zeros(len(clusters), dtype=np.uint32)
    indices = []

    for cluster in clusters:
        cluster_indices = np.where(input_array == cluster)[0]
        indptr[cluster - 1] = len(indices)
        indices.extend(cluster_indices)

    indices.extend(-np.zeros(len(input_array[input_array == 0])))
    indices = np.array(indices, dtype=np.int32).astype(np.uint32)
    return indices, indptr


def generate_experiment_file(
    template_path: Path,
    output_path: Path,
    cells_name: str = "seg_cells",
    analysis_name: str = "seg_analysis",
) -> None:
    """Generate Xenium experiment manifest file.

    Parameters
    ----------
    template_path : Path
        Path to template experiment.xenium file.
    output_path : Path
        Path for output experiment file.
    cells_name : str
        Name of cells Zarr file (without extension).
    analysis_name : str
        Name of analysis Zarr file (without extension).
    """
    with open(template_path) as f:
        experiment = json.load(f)

    experiment["xenium_explorer_files"]["cells_zarr_filepath"] = f"{cells_name}.zarr.zip"
    experiment["xenium_explorer_files"].pop("cell_features_zarr_filepath", None)
    experiment["xenium_explorer_files"]["analysis_zarr_filepath"] = f"{analysis_name}.zarr.zip"

    with open(output_path, "w") as f:
        json.dump(experiment, f, indent=2)


def seg2explorer(
    seg_df: Union[pd.DataFrame, pl.DataFrame],
    source_path: Union[str, Path],
    output_dir: Union[str, Path],
    cells_filename: str = "seg_cells",
    analysis_filename: str = "seg_analysis",
    xenium_filename: str = "seg_experiment.xenium",
    analysis_df: Optional[pd.DataFrame] = None,
    cell_id_column: str = "seg_cell_id",
    x_column: str = "x",
    y_column: str = "y",
    z_column: Optional[str] = "z",
    nucleus_column: Optional[str] = "cell_compartment",
    nucleus_value: int = 2,
    area_low: float = 10,
    area_high: float = 100,
) -> None:
    """Convert segmentation results to Xenium Explorer format.

    Parameters
    ----------
    seg_df : Union[pd.DataFrame, pl.DataFrame]
        Segmented transcript DataFrame with cell assignments.
    source_path : Union[str, Path]
        Path to source Xenium data directory.
    output_dir : Union[str, Path]
        Output directory for Zarr files.
    cells_filename : str
        Filename prefix for cells Zarr.
    analysis_filename : str
        Filename prefix for analysis Zarr.
    xenium_filename : str
        Filename for experiment manifest.
    analysis_df : Optional[pd.DataFrame]
        Optional clustering/annotation DataFrame.
    cell_id_column : str
        Column name for cell IDs.
    x_column : str
        Column name for x coordinates.
    y_column : str
        Column name for y coordinates.
    z_column : Optional[str]
        Column name for z coordinates (if available).
    nucleus_column : Optional[str]
        Column name for nucleus/compartment assignment.
    nucleus_value : int
        Value indicating nuclear compartment.
    area_low : float
        Minimum cell area threshold.
    area_high : float
        Maximum cell area threshold.
    """
    # Convert Polars to pandas
    if isinstance(seg_df, pl.DataFrame):
        seg_df = seg_df.to_pandas()

    source_path = Path(source_path)
    storage = Path(output_dir)
    storage.mkdir(parents=True, exist_ok=True)

    cell_id2old_id: Dict[int, Any] = {}
    cell_id: List[int] = []
    cell_summary: List[Dict[str, Any]] = []
    polygon_num_vertices: List[List[int]] = [[], []]
    polygon_vertices: List[List[Any]] = [[], []]
    seg_mask_value: List[int] = []

    grouped_by = seg_df.groupby(cell_id_column)

    for cell_incremental_id, (seg_cell_id, seg_cell) in tqdm(
        enumerate(grouped_by), total=len(grouped_by), desc="Processing cells"
    ):
        if len(seg_cell) < 5:
            continue

        cell_convex_hull = generate_boundary(seg_cell, x=x_column, y=y_column)
        if cell_convex_hull is None or not isinstance(cell_convex_hull, Polygon):
            continue

        if not (area_low <= cell_convex_hull.area <= area_high):
            continue

        uint_cell_id = cell_incremental_id + 1
        cell_id2old_id[uint_cell_id] = seg_cell_id

        # Handle nucleus if compartment info available
        nucleus_convex_hull = None
        if nucleus_column is not None and nucleus_column in seg_cell.columns:
            seg_nucleus = seg_cell[seg_cell[nucleus_column] == nucleus_value]
            if len(seg_nucleus) >= 3:
                try:
                    nucleus_convex_hull = ConvexHull(seg_nucleus[[x_column, y_column]])
                except Exception:
                    pass

        cell_id.append(uint_cell_id)

        # Compute z-level if available
        z_level = 0.0
        if z_column is not None and z_column in seg_cell.columns:
            z_level = (seg_cell[z_column].mean() // 3) * 3

        cell_summary.append({
            "cell_centroid_x": seg_cell[x_column].mean(),
            "cell_centroid_y": seg_cell[y_column].mean(),
            "cell_area": cell_convex_hull.area,
            "nucleus_centroid_x": seg_cell[x_column].mean(),
            "nucleus_centroid_y": seg_cell[y_column].mean(),
            "nucleus_area": cell_convex_hull.area,
            "z_level": z_level,
        })

        polygon_num_vertices[0].append(len(cell_convex_hull.exterior.coords))
        polygon_num_vertices[1].append(
            len(nucleus_convex_hull.vertices) if nucleus_convex_hull else 0
        )
        polygon_vertices[0].append(list(cell_convex_hull.exterior.coords))
        polygon_vertices[1].append(
            seg_nucleus[[x_column, y_column]].values[nucleus_convex_hull.vertices]
            if nucleus_convex_hull else np.array([[], []]).T
        )
        seg_mask_value.append(uint_cell_id)

    if len(cell_id) == 0:
        raise ValueError("No valid cells found in segmentation data.")

    cell_polygon_vertices = get_flatten_version(polygon_vertices[0], max_value=128)

    # Open source store and create new store
    source_zarr_store = ZipStore(source_path / "cells.zarr.zip", mode="r")
    existing_store = zarr.open(source_zarr_store, mode="r")
    new_store = zarr.open(storage / f"{cells_filename}.zarr.zip", mode="w")

    # Create polygon_sets group
    polygon_group = new_store.create_group("polygon_sets")
    cell_num_vertices = polygon_num_vertices[1]
    n_cells = cell_polygon_vertices.shape[0]
    cell_vertices_flat = cell_polygon_vertices.reshape(n_cells, -1)[:, :257]

    set1 = polygon_group.create_group("1")
    set1["cell_index"] = np.arange(1, n_cells + 1, dtype=np.uint32)
    set1["method"] = np.ones(n_cells, dtype=np.uint32)
    set1["num_vertices"] = np.array(polygon_num_vertices[0], dtype=np.int32)
    set1["vertices"] = cell_vertices_flat.astype(np.float32)

    new_store.attrs.update(existing_store.attrs)
    new_store.attrs["number_cells"] = len(cell_id)
    new_store.store.close()
    source_zarr_store.close()

    # Create analysis data
    if analysis_df is None:
        analysis_df = pd.DataFrame(
            [cell_id2old_id[i] for i in cell_id], columns=[cell_id_column]
        )
        analysis_df["default"] = "segger"

    zarr_df = pd.DataFrame(
        [cell_id2old_id[i] for i in cell_id], columns=[cell_id_column]
    )
    clustering_df = pd.merge(zarr_df, analysis_df, how="left", on=cell_id_column)
    clusters_names = [col for col in analysis_df.columns if col != cell_id_column]

    clusters_dict = {
        cluster: {
            label: idx + 1
            for idx, label in enumerate(sorted(np.unique(clustering_df[cluster].dropna())))
        }
        for cluster in clusters_names
    }

    new_zarr = zarr.open(storage / f"{analysis_filename}.zarr.zip", mode="w")
    new_zarr.create_group("/cell_groups")

    for i, cluster in enumerate(clusters_names):
        new_zarr["cell_groups"].create_group(str(i))
        group_values = [clusters_dict[cluster].get(x, 0) for x in clustering_df[cluster]]
        indices, indptr = get_indices_indptr(np.array(group_values))
        new_zarr["cell_groups"][str(i)]["indices"] = indices
        new_zarr["cell_groups"][str(i)]["indptr"] = indptr

    new_zarr["cell_groups"].attrs.update({
        "major_version": 1,
        "minor_version": 0,
        "number_groupings": len(clusters_names),
        "grouping_names": clusters_names,
        "group_names": [
            sorted(clusters_dict[cluster], key=clusters_dict[cluster].get)
            for cluster in clusters_names
        ],
    })
    new_zarr.store.close()

    generate_experiment_file(
        template_path=source_path / "experiment.xenium",
        output_path=storage / xenium_filename,
        cells_name=cells_filename,
        analysis_name=analysis_filename,
    )


def _process_one_cell(args: tuple) -> Optional[dict]:
    """Process a single cell for parallel boundary generation.

    Parameters
    ----------
    args : tuple
        Tuple of (seg_cell_id, seg_cell, x_col, y_col, area_low, area_high).

    Returns
    -------
    Optional[dict]
        Cell data dict or None if invalid.
    """
    seg_cell_id, seg_cell, x_col, y_col, area_low, area_high = args

    if len(seg_cell) < 5:
        return None

    cell_convex_hull = generate_boundary(seg_cell, x=x_col, y=y_col)
    if cell_convex_hull is None or not isinstance(cell_convex_hull, Polygon):
        return None

    if not (area_low <= cell_convex_hull.area <= area_high):
        return None

    # Get vertices and remove duplicate closing vertex
    cell_vertices = list(cell_convex_hull.exterior.coords)
    if cell_vertices[0] == cell_vertices[-1]:
        cell_vertices = cell_vertices[:-1]

    n_vertices = len(cell_vertices)

    # Sample up to 16 vertices
    if n_vertices > 16:
        indices = np.linspace(0, n_vertices - 1, 16, dtype=int)
        sampled_vertices = [cell_vertices[i] for i in indices]
    else:
        sampled_vertices = cell_vertices

    # Pad with first vertex if needed
    if len(sampled_vertices) < 16:
        sampled_vertices += [sampled_vertices[0]] * (16 - len(sampled_vertices))

    return {
        "seg_cell_id": seg_cell_id,
        "cell_area": float(cell_convex_hull.area),
        "cell_vertices": sampled_vertices,
        "cell_num_vertices": len(sampled_vertices),
    }


def seg2explorer_pqdm(
    seg_df: Union[pd.DataFrame, pl.DataFrame],
    source_path: Union[str, Path],
    output_dir: Union[str, Path],
    cells_filename: str = "seg_cells",
    analysis_filename: str = "seg_analysis",
    xenium_filename: str = "seg_experiment.xenium",
    analysis_df: Optional[pd.DataFrame] = None,
    cell_id_column: str = "seg_cell_id",
    x_column: str = "x",
    y_column: str = "y",
    area_low: float = 10,
    area_high: float = 100,
    n_jobs: int = 1,
) -> None:
    """Parallelized version of seg2explorer using pqdm.

    Parameters
    ----------
    seg_df : Union[pd.DataFrame, pl.DataFrame]
        Segmented transcript DataFrame.
    source_path : Union[str, Path]
        Path to source Xenium data.
    output_dir : Union[str, Path]
        Output directory.
    cells_filename : str
        Cells Zarr filename prefix.
    analysis_filename : str
        Analysis Zarr filename prefix.
    xenium_filename : str
        Experiment manifest filename.
    analysis_df : Optional[pd.DataFrame]
        Optional clustering annotations.
    cell_id_column : str
        Cell ID column name.
    x_column : str
        X coordinate column name.
    y_column : str
        Y coordinate column name.
    area_low : float
        Minimum cell area.
    area_high : float
        Maximum cell area.
    n_jobs : int
        Number of parallel workers.
    """
    # Convert Polars to pandas
    if isinstance(seg_df, pl.DataFrame):
        seg_df = seg_df.to_pandas()

    source_path = Path(source_path)
    storage = Path(output_dir)
    storage.mkdir(parents=True, exist_ok=True)

    grouped_by = seg_df.groupby(cell_id_column)

    # Build work items
    work_iter = (
        (seg_cell_id, seg_cell, x_column, y_column, area_low, area_high)
        for seg_cell_id, seg_cell in grouped_by
    )

    # Parallel processing
    results = pqdm(
        work_iter,
        _process_one_cell,
        n_jobs=n_jobs,
        desc="Processing cells",
        exception_behaviour="immediate",
    )

    # Collate results
    cell_id2old_id: Dict[int, Any] = {}
    cell_id: List[int] = []
    polygon_num_vertices: List[int] = []
    polygon_vertices: List[List[Any]] = []

    kept = [r for r in results if r is not None]
    for cell_incremental_id, r in enumerate(kept):
        uint_cell_id = cell_incremental_id + 1
        cell_id2old_id[uint_cell_id] = r["seg_cell_id"]
        cell_id.append(uint_cell_id)
        polygon_num_vertices.append(r["cell_num_vertices"])
        polygon_vertices.append(r["cell_vertices"])

    if len(cell_id) == 0:
        raise ValueError("No valid cells found in segmentation data.")

    cell_polygon_vertices = get_flatten_version(polygon_vertices)

    # Open source and create new store
    source_zarr_store = ZipStore(source_path / "cells.zarr.zip", mode="r")
    existing_store = zarr.open(source_zarr_store, mode="r")
    new_store = zarr.open(storage / f"{cells_filename}.zarr.zip", mode="w")

    polygon_group = new_store.create_group("polygon_sets")
    n_cells = cell_polygon_vertices.shape[0]
    cell_vertices_flat = cell_polygon_vertices.reshape(n_cells, -1)[:, :33]

    set1 = polygon_group.create_group("1")
    set1["cell_index"] = np.arange(1, n_cells + 1, dtype=np.uint32)
    set1["method"] = np.ones(n_cells, dtype=np.uint32)
    set1["num_vertices"] = np.array(polygon_num_vertices, dtype=np.int32)
    set1["vertices"] = cell_vertices_flat.astype(np.float32)

    new_store.attrs.update(existing_store.attrs)
    new_store.attrs["number_cells"] = n_cells
    new_store.store.close()
    source_zarr_store.close()

    # Create analysis data
    if analysis_df is None:
        analysis_df = pd.DataFrame(
            [cell_id2old_id[i] for i in cell_id], columns=[cell_id_column]
        )
        analysis_df["default"] = "segger"

    zarr_df = pd.DataFrame(
        [cell_id2old_id[i] for i in cell_id], columns=[cell_id_column]
    )
    clustering_df = pd.merge(zarr_df, analysis_df, how="left", on=cell_id_column)
    clusters_names = [col for col in analysis_df.columns if col != cell_id_column]

    clusters_dict = {
        cluster: {
            label: idx + 1
            for idx, label in enumerate(sorted(np.unique(clustering_df[cluster].dropna())))
        }
        for cluster in clusters_names
    }

    new_zarr = zarr.open(storage / f"{analysis_filename}.zarr.zip", mode="w")
    new_zarr.create_group("/cell_groups")

    for i, cluster in enumerate(clusters_names):
        new_zarr["cell_groups"].create_group(str(i))
        group_values = [clusters_dict[cluster].get(x, 0) for x in clustering_df[cluster]]
        indices, indptr = get_indices_indptr(np.array(group_values))
        new_zarr["cell_groups"][str(i)]["indices"] = indices
        new_zarr["cell_groups"][str(i)]["indptr"] = indptr

    new_zarr["cell_groups"].attrs.update({
        "major_version": 1,
        "minor_version": 0,
        "number_groupings": len(clusters_names),
        "grouping_names": clusters_names,
        "group_names": [
            sorted(clusters_dict[cluster], key=clusters_dict[cluster].get)
            for cluster in clusters_names
        ],
    })
    new_zarr.store.close()

    generate_experiment_file(
        template_path=source_path / "experiment.xenium",
        output_path=storage / xenium_filename,
        cells_name=cells_filename,
        analysis_name=analysis_filename,
    )
