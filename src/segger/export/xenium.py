"""Xenium Explorer export functionality.

This module converts segmentation results into Xenium Explorer-compatible
Zarr format for visualization and validation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd
import polars as pl
import zarr
from pqdm.processes import pqdm as pqdm_processes
try:
    from pqdm.threads import pqdm as pqdm_threads
except Exception:
    pqdm_threads = None
from shapely.geometry import MultiPoint, MultiPolygon, Polygon
from tqdm import tqdm
from zarr.storage import ZipStore

from .boundary import extract_largest_polygon, generate_boundary


def _normalize_polygon_vertices(
    polygon: Polygon,
    max_vertices: int,
) -> Tuple[List[Tuple[float, float]], int]:
    """Normalize polygon vertices to a fixed length with closure.

    Returns a list of vertices padded/truncated to ``max_vertices`` and the
    true number of vertices including the closing vertex.
    """
    coords = list(polygon.exterior.coords)
    # Remove duplicate closing vertex
    if coords[0] == coords[-1]:
        coords = coords[:-1]

    if len(coords) < 3:
        return [], 0

    num_vertices = len(coords) + 1  # include closing vertex
    target = max_vertices - 1

    if len(coords) > target:
        indices = np.linspace(0, len(coords) - 1, target, dtype=int)
        coords = [coords[i] for i in indices]

    # Close polygon and pad
    coords.append(coords[0])
    if len(coords) < max_vertices:
        coords += [coords[0]] * (max_vertices - len(coords))

    return coords, num_vertices


def _safe_boundary_polygon(
    seg_cell: pd.DataFrame,
    x: str,
    y: str,
    boundary_method: str = "delaunay",
    boundary_voxel_size: float = 0.0,
) -> Optional[Polygon]:
    """Generate a robust polygon boundary for a cell.

    Uses the requested boundary method with robust fallbacks.
    """
    if boundary_method in {"convex_hull", "input"}:
        mp = MultiPoint(seg_cell[[x, y]].values)
        cell_poly = mp.convex_hull if not mp.is_empty else None
    elif boundary_method == "voxel":
        if boundary_voxel_size <= 0:
            return None
        points = seg_cell[[x, y]].to_numpy(dtype=np.float64)
        if len(points) < 3:
            return None
        mins = points.min(axis=0)
        bins = np.floor((points - mins) / boundary_voxel_size).astype(np.int64)
        _, keep = np.unique(bins, axis=0, return_index=True)
        reduced = points[np.sort(keep)]
        if len(reduced) < 3:
            return None
        mp = MultiPoint(reduced)
        cell_poly = mp.convex_hull if not mp.is_empty else None
    else:
        working = seg_cell
        if boundary_voxel_size > 0:
            points = seg_cell[[x, y]].to_numpy(dtype=np.float64)
            mins = points.min(axis=0)
            bins = np.floor((points - mins) / boundary_voxel_size).astype(np.int64)
            _, keep = np.unique(bins, axis=0, return_index=True)
            working = seg_cell.iloc[np.sort(keep)]

        try:
            cell_poly = generate_boundary(working, x=x, y=y)
            if isinstance(cell_poly, MultiPolygon):
                cell_poly = extract_largest_polygon(cell_poly)
        except Exception:
            cell_poly = None

    if cell_poly is None or not isinstance(cell_poly, Polygon) or cell_poly.is_empty:
        # Fallback: convex hull of points
        mp = MultiPoint(seg_cell[[x, y]].values)
        cell_poly = mp.convex_hull if not mp.is_empty else None

    if cell_poly is None or not isinstance(cell_poly, Polygon) or cell_poly.is_empty:
        return None

    return cell_poly


def _prepare_input_boundaries(
    boundaries,
    boundary_id_column: str = "cell_id",
    boundary_type_column: str = "boundary_type",
    boundary_cell_value: str = "cell",
    boundary_nucleus_value: str = "nucleus",
) -> Tuple[Dict[Any, Polygon], Dict[Any, Polygon]]:
    """Prepare lookup tables for input cell/nucleus boundaries."""
    if boundaries is None:
        return {}, {}

    gdf = boundaries
    if boundary_id_column not in gdf.columns:
        if gdf.index.name == boundary_id_column:
            gdf = gdf.reset_index()
        else:
            return {}, {}

    def _pick_largest(group):
        largest = None
        max_area = -1.0
        for geom in group.geometry:
            if geom is None or getattr(geom, "is_empty", True):
                continue
            if isinstance(geom, MultiPolygon):
                geom = extract_largest_polygon(geom)
            if not isinstance(geom, Polygon) or geom is None or geom.is_empty:
                continue
            area = float(geom.area)
            if area > max_area:
                max_area = area
                largest = geom
        return largest

    if boundary_type_column in gdf.columns:
        cells = gdf[gdf[boundary_type_column] == boundary_cell_value]
        nuclei = gdf[gdf[boundary_type_column] == boundary_nucleus_value]
    else:
        cells = gdf
        nuclei = gdf.iloc[0:0]

    cell_lookup: Dict[Any, Polygon] = {}
    for cell_id, group in cells.groupby(boundary_id_column):
        poly = _pick_largest(group)
        if poly is not None:
            cell_lookup[cell_id] = poly

    nucleus_lookup: Dict[Any, Polygon] = {}
    for cell_id, group in nuclei.groupby(boundary_id_column):
        poly = _pick_largest(group)
        if poly is not None:
            nucleus_lookup[cell_id] = poly

    return cell_lookup, nucleus_lookup


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
    Notes
    -----
    We only replace the cells and analysis Zarr paths, preserving all other
    entries (including morphology image references). This keeps multi-channel
    morphology_focus image stacks intact for segmentation kit datasets.
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
    polygon_max_vertices: int = 13,
    boundary_method: str = "delaunay",
    boundary_voxel_size: float = 0.0,
    boundaries: Optional["gpd.GeoDataFrame"] = None,
    boundary_id_column: str = "cell_id",
    boundary_type_column: str = "boundary_type",
    boundary_cell_value: str = "cell",
    boundary_nucleus_value: str = "nucleus",
    cell_id_columns: Optional[str] = None,
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
    polygon_max_vertices : int
        Maximum number of vertices per polygon (including closure).
    """
    if cell_id_columns is not None:
        cell_id_column = cell_id_columns

    if boundary_method == "skip":
        raise ValueError("boundary_method='skip' is not supported for Xenium export.")

    # Convert Polars to pandas
    if isinstance(seg_df, pl.DataFrame):
        seg_df = seg_df.to_pandas()

    source_path = Path(source_path)
    storage = Path(output_dir)
    storage.mkdir(parents=True, exist_ok=True)

    cell_boundaries: Dict[Any, Polygon] = {}
    nucleus_boundaries: Dict[Any, Polygon] = {}
    if boundary_method == "input":
        cell_boundaries, nucleus_boundaries = _prepare_input_boundaries(
            boundaries=boundaries,
            boundary_id_column=boundary_id_column,
            boundary_type_column=boundary_type_column,
            boundary_cell_value=boundary_cell_value,
            boundary_nucleus_value=boundary_nucleus_value,
        )

    # Drop unassigned cells if numeric
    if cell_id_column in seg_df.columns:
        if pd.api.types.is_numeric_dtype(seg_df[cell_id_column]):
            seg_df = seg_df[seg_df[cell_id_column] >= 0]
        else:
            seg_df = seg_df[seg_df[cell_id_column].notna()]

    cell_id2old_id: Dict[int, Any] = {}
    cell_id: List[int] = []
    cell_summary_rows: List[List[float]] = []
    cell_num_vertices: List[int] = []
    nucleus_num_vertices: List[int] = []
    cell_vertices: List[List[Tuple[float, float]]] = []
    nucleus_vertices: List[List[Tuple[float, float]]] = []

    grouped_by = seg_df.groupby(cell_id_column)

    for cell_incremental_id, (seg_cell_id, seg_cell) in tqdm(
        enumerate(grouped_by), total=len(grouped_by), desc="Processing cells"
    ):
        if len(seg_cell) < 5:
            continue

        if boundary_method == "input" and cell_boundaries:
            cell_poly = cell_boundaries.get(seg_cell_id)
        else:
            fallback_method = "delaunay" if boundary_method == "input" else boundary_method
            cell_poly = _safe_boundary_polygon(
                seg_cell,
                x=x_column,
                y=y_column,
                boundary_method=fallback_method,
                boundary_voxel_size=boundary_voxel_size,
            )
        if cell_poly is None or not (area_low <= cell_poly.area <= area_high):
            continue

        # Nucleus polygon (optional)
        nucleus_poly = None
        if boundary_method == "input" and nucleus_boundaries:
            nucleus_poly = nucleus_boundaries.get(seg_cell_id)
        elif nucleus_column is not None and nucleus_column in seg_cell.columns:
            seg_nucleus = seg_cell[seg_cell[nucleus_column] == nucleus_value]
            if len(seg_nucleus) >= 3:
                nucleus_poly = MultiPoint(seg_nucleus[[x_column, y_column]].values).convex_hull
                if isinstance(nucleus_poly, MultiPolygon):
                    nucleus_poly = extract_largest_polygon(nucleus_poly)
                if not isinstance(nucleus_poly, Polygon) or nucleus_poly.is_empty:
                    nucleus_poly = None

        cell_coords, cell_nv = _normalize_polygon_vertices(cell_poly, polygon_max_vertices)
        if cell_nv == 0:
            continue

        zero_vertices = [(0.0, 0.0)] * polygon_max_vertices
        if nucleus_poly is not None:
            nuc_coords, nuc_nv = _normalize_polygon_vertices(nucleus_poly, polygon_max_vertices)
        else:
            nuc_coords, nuc_nv = zero_vertices, 0

        uint_cell_id = cell_incremental_id + 1
        cell_id2old_id[uint_cell_id] = seg_cell_id
        cell_id.append(uint_cell_id)

        # Compute z-level if available
        z_level = 0.0
        if z_column is not None and z_column in seg_cell.columns:
            z_level = (seg_cell[z_column].mean() // 3) * 3

        cell_centroid = cell_poly.centroid
        nucleus_centroid = nucleus_poly.centroid if nucleus_poly is not None else None

        cell_summary_rows.append([
            float(cell_centroid.x),
            float(cell_centroid.y),
            float(cell_poly.area),
            float(nucleus_centroid.x) if nucleus_centroid is not None else 0.0,
            float(nucleus_centroid.y) if nucleus_centroid is not None else 0.0,
            float(nucleus_poly.area) if nucleus_poly is not None else 0.0,
            float(z_level),
            float(1 if nucleus_poly is not None else 0),
        ])

        cell_num_vertices.append(cell_nv)
        nucleus_num_vertices.append(nuc_nv)
        cell_vertices.append(cell_coords)
        nucleus_vertices.append(nuc_coords)

    if len(cell_id) == 0:
        raise ValueError("No valid cells found in segmentation data.")

    n_cells = len(cell_id)
    cell_vertices_arr = np.array(cell_vertices, dtype=np.float32)
    nucleus_vertices_arr = np.array(nucleus_vertices, dtype=np.float32)
    cell_vertices_flat = cell_vertices_arr.reshape(n_cells, -1)
    nucleus_vertices_flat = nucleus_vertices_arr.reshape(n_cells, -1)

    # Open source store and create new store
    source_zarr_store = ZipStore(source_path / "cells.zarr.zip", mode="r")
    existing_store = zarr.open(source_zarr_store, mode="r")
    new_store = zarr.open(storage / f"{cells_filename}.zarr.zip", mode="w")

    # Root datasets
    cell_id_arr = np.zeros((n_cells, 2), dtype=np.uint32)
    cell_id_arr[:, 1] = np.array(cell_id, dtype=np.uint32)
    new_store["cell_id"] = cell_id_arr
    new_store["cell_summary"] = np.array(cell_summary_rows, dtype=np.float64)

    # Polygon sets
    polygon_group = new_store.create_group("polygon_sets")

    # Nucleus polygons (set 0)
    set0 = polygon_group.create_group("0")
    set0["cell_index"] = np.array(cell_id, dtype=np.uint32)
    set0["method"] = np.zeros(n_cells, dtype=np.uint32)
    set0["num_vertices"] = np.array(nucleus_num_vertices, dtype=np.int32)
    set0["vertices"] = nucleus_vertices_flat.astype(np.float32)

    # Cell polygons (set 1)
    set1 = polygon_group.create_group("1")
    set1["cell_index"] = np.array(cell_id, dtype=np.uint32)
    set1["method"] = np.full(n_cells, 1, dtype=np.uint32)
    set1["num_vertices"] = np.array(cell_num_vertices, dtype=np.int32)
    set1["vertices"] = cell_vertices_flat.astype(np.float32)

    # Update attributes
    attrs = dict(existing_store.attrs)
    attrs["number_cells"] = n_cells
    attrs["polygon_set_names"] = ["nucleus", "cell"]
    attrs["polygon_set_display_names"] = ["Nucleus", "Cell"]
    attrs["polygon_set_descriptions"] = [
        "Segger nucleus boundaries",
        "Segger cell boundaries",
    ]
    cell_method = f"segger_cell_{boundary_method}"
    nucleus_method = "segger_nucleus_convex_hull"
    if boundary_method == "input" and nucleus_boundaries:
        nucleus_method = "segger_nucleus_input"
    attrs["segmentation_methods"] = [nucleus_method, cell_method]
    attrs.setdefault("spatial_units", "microns")
    attrs.setdefault("major_version", 4)
    attrs.setdefault("minor_version", 0)
    new_store.attrs.update(attrs)

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
    """Process a single cell for parallel boundary generation."""
    (
        seg_cell_id,
        seg_cell,
        x_col,
        y_col,
        z_col,
        nucleus_column,
        nucleus_value,
        area_low,
        area_high,
        polygon_max_vertices,
        boundary_method,
        boundary_voxel_size,
    ) = args

    if len(seg_cell) < 5:
        return None

    cell_poly = _safe_boundary_polygon(
        seg_cell,
        x=x_col,
        y=y_col,
        boundary_method=boundary_method,
        boundary_voxel_size=boundary_voxel_size,
    )
    if cell_poly is None or not (area_low <= cell_poly.area <= area_high):
        return None

    cell_vertices, cell_nv = _normalize_polygon_vertices(cell_poly, polygon_max_vertices)
    if cell_nv == 0:
        return None

    # Nucleus polygon (optional)
    nucleus_poly = None
    if nucleus_column is not None and nucleus_column in seg_cell.columns:
        seg_nucleus = seg_cell[seg_cell[nucleus_column] == nucleus_value]
        if len(seg_nucleus) >= 3:
            nucleus_poly = MultiPoint(seg_nucleus[[x_col, y_col]].values).convex_hull
            if isinstance(nucleus_poly, MultiPolygon):
                nucleus_poly = extract_largest_polygon(nucleus_poly)
            if not isinstance(nucleus_poly, Polygon) or nucleus_poly.is_empty:
                nucleus_poly = None

    if nucleus_poly is not None:
        nucleus_vertices, nucleus_nv = _normalize_polygon_vertices(
            nucleus_poly, polygon_max_vertices
        )
    else:
        nucleus_vertices = [(0.0, 0.0)] * polygon_max_vertices
        nucleus_nv = 0

    # Compute z-level if available
    z_level = 0.0
    if z_col is not None and z_col in seg_cell.columns:
        z_level = (seg_cell[z_col].mean() // 3) * 3

    cell_centroid = cell_poly.centroid
    nucleus_centroid = nucleus_poly.centroid if nucleus_poly is not None else None

    return {
        "seg_cell_id": seg_cell_id,
        "cell_area": float(cell_poly.area),
        "cell_vertices": cell_vertices,
        "cell_num_vertices": cell_nv,
        "nucleus_vertices": nucleus_vertices,
        "nucleus_num_vertices": nucleus_nv,
        "cell_centroid_x": float(cell_centroid.x),
        "cell_centroid_y": float(cell_centroid.y),
        "nucleus_centroid_x": float(nucleus_centroid.x) if nucleus_centroid else 0.0,
        "nucleus_centroid_y": float(nucleus_centroid.y) if nucleus_centroid else 0.0,
        "nucleus_area": float(nucleus_poly.area) if nucleus_poly is not None else 0.0,
        "z_level": float(z_level),
        "nucleus_count": float(1 if nucleus_poly is not None else 0),
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
    z_column: Optional[str] = "z",
    nucleus_column: Optional[str] = "cell_compartment",
    nucleus_value: int = 2,
    area_low: float = 10,
    area_high: float = 100,
    n_jobs: int = 1,
    polygon_max_vertices: int = 13,
    boundary_method: str = "delaunay",
    boundary_voxel_size: float = 0.0,
    boundaries: Optional["gpd.GeoDataFrame"] = None,
    boundary_id_column: str = "cell_id",
    boundary_type_column: str = "boundary_type",
    boundary_cell_value: str = "cell",
    boundary_nucleus_value: str = "nucleus",
    cell_id_columns: Optional[str] = None,
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
    z_column : Optional[str]
        Z coordinate column name (if available).
    nucleus_column : Optional[str]
        Column name for nucleus/compartment assignment.
    nucleus_value : int
        Value indicating nuclear compartment.
    area_low : float
        Minimum cell area.
    area_high : float
        Maximum cell area.
    n_jobs : int
        Number of parallel workers.
    polygon_max_vertices : int
        Maximum number of vertices per polygon (including closure).
    """
    if cell_id_columns is not None:
        cell_id_column = cell_id_columns

    if boundary_method == "skip":
        raise ValueError("boundary_method='skip' is not supported for Xenium export.")
    if boundary_method == "input" and boundaries is not None:
        raise ValueError(
            "Parallel Xenium export does not support boundary_method='input'. "
            "Use seg2explorer (serial) when passing input boundaries."
        )
    if boundary_method == "input":
        boundary_method = "delaunay"

    # Convert Polars to pandas
    if isinstance(seg_df, pl.DataFrame):
        seg_df = seg_df.to_pandas()

    source_path = Path(source_path)
    storage = Path(output_dir)
    storage.mkdir(parents=True, exist_ok=True)

    grouped_by = seg_df.groupby(cell_id_column)

    def _work_iter():
        return (
            (
                seg_cell_id,
                seg_cell,
                x_column,
                y_column,
                z_column,
                nucleus_column,
                nucleus_value,
                area_low,
                area_high,
                polygon_max_vertices,
                boundary_method,
                boundary_voxel_size,
            )
            for seg_cell_id, seg_cell in grouped_by
        )

    # Process backend first for throughput and "whole job" progress visibility.
    # If the process pool crashes, restart once with thread workers.
    try:
        results = pqdm_processes(
            _work_iter(),
            _process_one_cell,
            n_jobs=n_jobs,
            desc="Processing cells",
            exception_behaviour="immediate",
        )
    except BrokenProcessPool:
        if pqdm_threads is None:
            raise RuntimeError(
                "Process workers crashed and pqdm thread backend is unavailable."
            )
        tqdm.write(
            "Warning: process workers crashed during Xenium export. "
            "Retrying with thread workers from 0% (completed process results "
            "cannot be recovered by pqdm)."
        )
        results = pqdm_threads(
            _work_iter(),
            _process_one_cell,
            n_jobs=n_jobs,
            desc="Processing cells (thread fallback)",
            exception_behaviour="immediate",
        )

    # Collate results
    cell_id2old_id: Dict[int, Any] = {}
    cell_id: List[int] = []
    cell_num_vertices: List[int] = []
    nucleus_num_vertices: List[int] = []
    cell_vertices: List[List[Any]] = []
    nucleus_vertices: List[List[Any]] = []
    cell_summary_rows: List[List[float]] = []

    kept = [r for r in results if r is not None]
    for cell_incremental_id, r in enumerate(kept):
        uint_cell_id = cell_incremental_id + 1
        cell_id2old_id[uint_cell_id] = r["seg_cell_id"]
        cell_id.append(uint_cell_id)
        cell_num_vertices.append(r["cell_num_vertices"])
        nucleus_num_vertices.append(r["nucleus_num_vertices"])
        cell_vertices.append(r["cell_vertices"])
        nucleus_vertices.append(r["nucleus_vertices"])
        cell_summary_rows.append([
            r["cell_centroid_x"],
            r["cell_centroid_y"],
            r["cell_area"],
            r["nucleus_centroid_x"],
            r["nucleus_centroid_y"],
            r["nucleus_area"],
            r["z_level"],
            r["nucleus_count"],
        ])

    if len(cell_id) == 0:
        raise ValueError("No valid cells found in segmentation data.")

    n_cells = len(cell_id)
    cell_vertices_arr = np.array(cell_vertices, dtype=np.float32)
    nucleus_vertices_arr = np.array(nucleus_vertices, dtype=np.float32)
    cell_vertices_flat = cell_vertices_arr.reshape(n_cells, -1)
    nucleus_vertices_flat = nucleus_vertices_arr.reshape(n_cells, -1)

    # Open source and create new store
    source_zarr_store = ZipStore(source_path / "cells.zarr.zip", mode="r")
    existing_store = zarr.open(source_zarr_store, mode="r")
    new_store = zarr.open(storage / f"{cells_filename}.zarr.zip", mode="w")

    # Root datasets
    cell_id_arr = np.zeros((n_cells, 2), dtype=np.uint32)
    cell_id_arr[:, 1] = np.array(cell_id, dtype=np.uint32)
    new_store["cell_id"] = cell_id_arr
    new_store["cell_summary"] = np.array(cell_summary_rows, dtype=np.float64)

    polygon_group = new_store.create_group("polygon_sets")

    # Nucleus polygons (set 0)
    set0 = polygon_group.create_group("0")
    set0["cell_index"] = np.array(cell_id, dtype=np.uint32)
    set0["method"] = np.zeros(n_cells, dtype=np.uint32)
    set0["num_vertices"] = np.array(nucleus_num_vertices, dtype=np.int32)
    set0["vertices"] = nucleus_vertices_flat.astype(np.float32)

    # Cell polygons (set 1)
    set1 = polygon_group.create_group("1")
    set1["cell_index"] = np.array(cell_id, dtype=np.uint32)
    set1["method"] = np.full(n_cells, 1, dtype=np.uint32)
    set1["num_vertices"] = np.array(cell_num_vertices, dtype=np.int32)
    set1["vertices"] = cell_vertices_flat.astype(np.float32)

    attrs = dict(existing_store.attrs)
    attrs["number_cells"] = n_cells
    attrs["polygon_set_names"] = ["nucleus", "cell"]
    attrs["polygon_set_display_names"] = ["Nucleus", "Cell"]
    attrs["polygon_set_descriptions"] = [
        "Segger nucleus boundaries",
        "Segger cell boundaries",
    ]
    attrs["segmentation_methods"] = ["segger_nucleus_convex_hull", f"segger_cell_{boundary_method}"]
    attrs.setdefault("spatial_units", "microns")
    attrs.setdefault("major_version", 4)
    attrs.setdefault("minor_version", 0)
    new_store.attrs.update(attrs)
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
