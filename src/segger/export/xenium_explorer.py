import os
import sys
from pathlib import Path
import gzip
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPolygon, Polygon
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Tuple
from segger.export.boundary import generate_boundary as _generate_boundary_method
from segger.prediction.boundary import generate_boundary as _generate_boundary_delaunay
from zarr.storage import ZipStore
import zarr


def get_flatten_version(polygon_vertices: List[List[Tuple[float, float]]], max_value: int = 21) -> np.ndarray:
    """Standardize list of polygon vertices to a fixed shape.

    Args:
        polygon_vertices (List[List[Tuple[float, float]]]): List of polygon coordinate lists.
        max_value (int): Max number of coordinates per polygon.

    Returns:
        np.ndarray: Padded or truncated list of polygon vertices.
    """
    flattened = []
    for vertices in polygon_vertices:
        if len(vertices) < 3:
            pass
        if isinstance(vertices, np.ndarray):
            vertices = vertices.tolist()
        
        if len(vertices) > max_value:
            flattened.append(vertices[:max_value])
        else:
            flattened.append(vertices + [vertices[0]] * (max_value - len(vertices)))
    return np.array(flattened, dtype=np.float32)


def _boundary_from_method(seg_cell, boundary_method: str, boundary_voxel_size: float):
    if boundary_method == "convex_hull" or boundary_method == "input":
        points = seg_cell[["x_location", "y_location"]].values
        if len(points) < 3:
            return None
        try:
            hull = ConvexHull(points)
            return Polygon(points[hull.vertices])
        except Exception:
            return None
    if boundary_method == "delaunay":
        if boundary_voxel_size > 0:
            pts = seg_cell[["x_location", "y_location"]].to_numpy()
            mins = pts.min(axis=0)
            bins = np.floor((pts - mins) / boundary_voxel_size).astype(np.int64)
            _, keep = np.unique(bins, axis=0, return_index=True)
            if keep.size < len(seg_cell):
                seg_cell = seg_cell.iloc[keep]
        geom = _generate_boundary_delaunay(seg_cell, x="x_location", y="y_location")
    elif boundary_method == "voxel":
        if boundary_voxel_size <= 0:
            return None
        geom = _generate_boundary_method(
            seg_cell,
            x="x_location",
            y="y_location",
            method="voxel",
            voxel_size=boundary_voxel_size,
        )
    else:
        geom = None
    if geom is not None:
        if isinstance(geom, MultiPolygon):
            return max(geom.geoms, key=lambda p: p.area) if len(geom.geoms) > 0 else None
        return geom
    return None


def _prepare_input_boundaries(
    boundaries,
    boundary_id_column: str = "cell_id",
    boundary_type_column: str = "boundary_type",
    boundary_cell_value: str = "cell",
    boundary_nucleus_value: str = "nucleus",
):
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
                geom = max(geom.geoms, key=lambda p: p.area) if len(geom.geoms) > 0 else None
            if not isinstance(geom, Polygon) or geom is None or geom.is_empty:
                continue
            area = geom.area
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

    cell_lookup: dict[Any, Polygon] = {}
    for cell_id, group in cells.groupby(boundary_id_column):
        poly = _pick_largest(group)
        if poly is not None:
            cell_lookup[cell_id] = poly

    nucleus_lookup: dict[Any, Polygon] = {}
    for cell_id, group in nuclei.groupby(boundary_id_column):
        poly = _pick_largest(group)
        if poly is not None:
            nucleus_lookup[cell_id] = poly

    return cell_lookup, nucleus_lookup


def seg2explorer(
    seg_df: pd.DataFrame,
    source_path: str,
    output_dir: str,
    cells_filename: str = "seg_cells",
    analysis_filename: str = "seg_analysis",
    xenium_filename: str = "seg_experiment.xenium",
    analysis_df: Optional[pd.DataFrame] = None,
    draw: bool = False,
    cell_id_columns: str = "seg_cell_id",
    area_low: float = 10,
    area_high: float = 100,
    boundary_method: str = "convex_hull",
    boundary_voxel_size: float = 0.0,
    boundaries: Optional["gpd.GeoDataFrame"] = None,
    boundary_id_column: str = "cell_id",
    boundary_type_column: str = "boundary_type",
    boundary_cell_value: str = "cell",
    boundary_nucleus_value: str = "nucleus",
) -> None:
    """Convert segmentation results into a Xenium Explorer-compatible Zarr dataset.

    Args:
        seg_df (pd.DataFrame): Segmented transcript dataframe.
        source_path (str): Path to the original Zarr store.
        output_dir (str): Output directory to save new Zarr and Xenium files.
        cells_filename (str): Filename prefix for cell Zarr file.
        analysis_filename (str): Filename prefix for cell group Zarr file.
        xenium_filename (str): Output experiment filename for Xenium.
        analysis_df (Optional[pd.DataFrame]): Optional dataframe with cluster annotations.
        draw (bool): Whether to draw polygons (not used currently).
        cell_id_columns (str): Column containing cell IDs.
        area_low (float): Minimum area threshold to include cells.
        area_high (float): Maximum area threshold to include cells.
    """
    source_path = Path(source_path)
    storage = Path(output_dir)
    storage.mkdir(parents=True, exist_ok=True)

    cell_id2old_id: Dict[int, Any] = {}
    cell_id: List[int] = []
    cell_summary: List[Dict[str, Any]] = []
    polygon_num_vertices: List[List[int]] = [[], []]
    polygon_vertices: List[List[Any]] = [[], []]
    seg_mask_value: List[int] = []

    cell_boundaries = {}
    nucleus_boundaries = {}
    if boundary_method == "input":
        cell_boundaries, nucleus_boundaries = _prepare_input_boundaries(
            boundaries,
            boundary_id_column=boundary_id_column,
            boundary_type_column=boundary_type_column,
            boundary_cell_value=boundary_cell_value,
            boundary_nucleus_value=boundary_nucleus_value,
        )

    has_cell_ids = seg_df is not None and cell_id_columns in seg_df.columns
    grouped_by = seg_df.groupby(cell_id_columns) if has_cell_ids else []
    seen_cells = set()

    for cell_incremental_id, (seg_cell_id, seg_cell) in tqdm(
        enumerate(grouped_by), total=len(grouped_by)
    ):
        if len(seg_cell) < 5:
            continue

        if boundary_method == "input" and cell_boundaries:
            cell_convex_hull = cell_boundaries.get(seg_cell_id)
        else:
            cell_convex_hull = _boundary_from_method(
                seg_cell, boundary_method, boundary_voxel_size
            )
        if cell_convex_hull is None or not isinstance(cell_convex_hull, Polygon):
            continue

        if not (area_low <= cell_convex_hull.area <= area_high):
            continue

        uint_cell_id = cell_incremental_id + 1
        cell_id2old_id[uint_cell_id] = seg_cell_id

        nucleus_convex_hull = None
        if boundary_method == "input" and nucleus_boundaries:
            nucleus_convex_hull = nucleus_boundaries.get(seg_cell_id)
        else:
            seg_nucleous = seg_cell[seg_cell["overlaps_nucleus"] == 1]
            if len(seg_nucleous) >= 3:
                try:
                    nucleus_convex_hull = ConvexHull(seg_nucleous[["x_location", "y_location"]])
                except Exception:
                    pass

        cell_id.append(uint_cell_id)
        cell_summary.append(
            {
                "cell_centroid_x": seg_cell["x_location"].mean(),
                "cell_centroid_y": seg_cell["y_location"].mean(),
                "cell_area": cell_convex_hull.area,
                "nucleus_centroid_x": seg_cell["x_location"].mean(),
                "nucleus_centroid_y": seg_cell["y_location"].mean(),
                "nucleus_area": cell_convex_hull.area,
                "z_level": (seg_cell.z_location.mean() // 3).round(0) * 3,
            }
        )
        polygon_num_vertices[0].append(len(cell_convex_hull.exterior.coords))
        polygon_num_vertices[1].append(
            len(nucleus_convex_hull.vertices) if nucleus_convex_hull else 0
        )
        polygon_vertices[0].append(list(cell_convex_hull.exterior.coords))
        polygon_vertices[1].append(
            seg_nucleous[["x_location", "y_location"]].values[
                nucleus_convex_hull.vertices
            ]
            if nucleus_convex_hull else np.array([[], []]).T
        )
        seg_mask_value.append(uint_cell_id)
        seen_cells.add(seg_cell_id)

    if boundary_method == "input" and cell_boundaries:
        for seg_cell_id, cell_poly in cell_boundaries.items():
            if seg_cell_id in seen_cells:
                continue
            if cell_poly is None or not isinstance(cell_poly, Polygon):
                continue
            if not (area_low <= cell_poly.area <= area_high):
                continue

            uint_cell_id = len(cell_id) + 1
            cell_id2old_id[uint_cell_id] = seg_cell_id
            cell_id.append(uint_cell_id)

            nucleus_poly = nucleus_boundaries.get(seg_cell_id)
            cell_summary.append(
                {
                    "cell_centroid_x": cell_poly.centroid.x,
                    "cell_centroid_y": cell_poly.centroid.y,
                    "cell_area": cell_poly.area,
                    "nucleus_centroid_x": nucleus_poly.centroid.x if isinstance(nucleus_poly, Polygon) else cell_poly.centroid.x,
                    "nucleus_centroid_y": nucleus_poly.centroid.y if isinstance(nucleus_poly, Polygon) else cell_poly.centroid.y,
                    "nucleus_area": nucleus_poly.area if isinstance(nucleus_poly, Polygon) else 0.0,
                    "z_level": 0.0,
                }
            )
            polygon_num_vertices[0].append(len(cell_poly.exterior.coords))
            polygon_num_vertices[1].append(len(nucleus_poly.exterior.coords) if isinstance(nucleus_poly, Polygon) else 0)
            polygon_vertices[0].append(list(cell_poly.exterior.coords))
            polygon_vertices[1].append(
                np.array([[], []]).T if not isinstance(nucleus_poly, Polygon)
                else np.array(nucleus_poly.exterior.coords)
            )
            seg_mask_value.append(uint_cell_id)

    cell_polygon_vertices = get_flatten_version(polygon_vertices[0], max_value=128)
    # nucl_polygon_vertices = get_flatten_version(polygon_vertices[1], max_value=16)

    cells = {
        "cell_id": np.array(
            [np.array(cell_id), np.ones(len(cell_id))], dtype=np.uint32
        ).T,
        "cell_summary": pd.DataFrame(cell_summary).values.astype(np.float64),
        "polygon_num_vertices": np.array(
            [
                [min(x + 1, x + 1) for x in polygon_num_vertices[1]],
                [min(x + 1, x + 1) for x in polygon_num_vertices[0]],
            ],
            dtype=np.int32,
        ),
        # "polygon_vertices": np.array(
        #     [nucl_polygon_vertices, cell_polygon_vertices], dtype=np.float32
        # ),
        "seg_mask_value": np.array(seg_mask_value, dtype=np.int32),
    }

    source_zarr_store = ZipStore(source_path / "cells.zarr.zip", mode="r") # added this line
    existing_store = zarr.open(source_zarr_store, mode="r")
    output_cells_store = ZipStore(storage / f"{cells_filename}.zarr.zip", mode="w")
    try:
        new_store = zarr.open_group(output_cells_store, mode="w", zarr_format=2)
    except TypeError:
        new_store = zarr.open(output_cells_store, mode="w")

    # Create polygon_sets group with the new structure
    polygon_group = new_store.create_group("polygon_sets")

    # Process cell polygons (set 1)
    # cell_polygons = cells["polygon_vertices"][1]  # Cell polygons are at index 1
    cell_num_vertices = cells["polygon_num_vertices"][1]  # Cell vertex counts

    # Reshape cell polygons to (n_cells, 50) format
    n_cells = cell_polygon_vertices.shape[0]
    cell_vertices_flat = cell_polygon_vertices.reshape(n_cells, -1)[:, :257]  # Take first 50 values

    set1 = polygon_group.create_group("1")
    set1["cell_index"] = np.arange(1, n_cells + 1, dtype=np.uint32)  # 1-based indexing
    set1["method"] = np.ones(n_cells, dtype=np.uint32)  # All method=1
    set1["num_vertices"] = np.array(cell_num_vertices, dtype=np.int32)
    set1["vertices"] = cell_vertices_flat.astype(np.float32)

    new_store.attrs.update(existing_store.attrs)
    new_store.attrs["number_cells"] = len(cells["cell_id"])
    new_store.store.close()
    source_zarr_store.close()

    if analysis_df is None:
        analysis_df = pd.DataFrame(
            [cell_id2old_id[i] for i in cell_id], columns=[cell_id_columns]
        )
        analysis_df["default"] = "segegger"

    zarr_df = pd.DataFrame(
        [cell_id2old_id[i] for i in cell_id], columns=[cell_id_columns]
    )
    clustering_df = pd.merge(zarr_df, analysis_df, how="left", on=cell_id_columns)
    clusters_names = [col for col in analysis_df.columns if col != cell_id_columns]

    clusters_dict = {
        cluster: {
            label: idx + 1
            for idx, label in enumerate(
                sorted(np.unique(clustering_df[cluster].dropna()))
            )
        }
        for cluster in clusters_names
    }

    output_analysis_store = ZipStore(storage / f"{analysis_filename}.zarr.zip", mode="w")
    try:
        new_zarr = zarr.open_group(output_analysis_store, mode="w", zarr_format=2)
    except TypeError:
        new_zarr = zarr.open(output_analysis_store, mode="w")
    new_zarr.create_group("/cell_groups")
    for i, cluster in enumerate(clusters_names):
        new_zarr["cell_groups"].create_group(str(i))
        group_values = [clusters_dict[cluster].get(x, 0) for x in clustering_df[cluster]]
        indices, indptr = get_indices_indptr(np.array(group_values))
        new_zarr["cell_groups"][str(i)]["indices"] = indices
        new_zarr["cell_groups"][str(i)]["indptr"] = indptr

    new_zarr["cell_groups"].attrs.update(
        {
            "major_version": 1,
            "minor_version": 0,
            "number_groupings": len(clusters_names),
            "grouping_names": clusters_names,
            "group_names": [
                sorted(clusters_dict[cluster], key=clusters_dict[cluster].get)
                for cluster in clusters_names
            ],
        }
    )
    new_zarr.store.close()

    generate_experiment_file(
        template_path=source_path / "experiment.xenium",
        output_path=storage / xenium_filename,
        cells_name=cells_filename,
        analysis_name=analysis_filename,
    )



def str_to_uint32(cell_id_str: str) -> Tuple[int, int]:
    """Convert a string cell ID back to uint32 format.

    Args:
        cell_id_str (str): The cell ID in string format.

    Returns:
        Tuple[int, int]: The cell ID in uint32 format and the dataset suffix.
    """
    prefix, suffix = cell_id_str.split("-")
    str_to_hex_mapping = {
        "a": "0",
        "b": "1",
        "c": "2",
        "d": "3",
        "e": "4",
        "f": "5",
        "g": "6",
        "h": "7",
        "i": "8",
        "j": "9",
        "k": "a",
        "l": "b",
        "m": "c",
        "n": "d",
        "o": "e",
        "p": "f",
    }
    hex_prefix = "".join([str_to_hex_mapping[char] for char in prefix])
    cell_id_uint32 = int(hex_prefix, 16)
    dataset_suffix = int(suffix)
    return cell_id_uint32, dataset_suffix


def get_indices_indptr(input_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get the indices and indptr arrays for sparse matrix representation.

    Args:
        input_array (np.ndarray): The input array containing cluster labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The indices and indptr arrays.
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


def save_cell_clustering(
    merged: pd.DataFrame, zarr_path: str, columns: List[str]
) -> None:
    """Save cell clustering information to a Zarr file.

    Args:
        merged (pd.DataFrame): The merged dataframe containing cell clustering information.
        zarr_path (str): The path to the Zarr file.
        columns (List[str]): The list of columns to save.
    """
    import zarr

    new_zarr = zarr.open(zarr_path, mode="w")
    new_zarr.create_group("/cell_groups")

    mappings = []
    for index, column in enumerate(columns):
        new_zarr["cell_groups"].create_group(index)
        classes = list(np.unique(merged[column].astype(str)))
        mapping_dict = {
            key: i
            for i, key in zip(
                range(1, len(classes)), [k for k in classes if k != "nan"]
            )
        }
        mapping_dict["nan"] = 0

        clusters = merged[column].astype(str).replace(mapping_dict).values.astype(int)
        indices, indptr = get_indices_indptr(clusters)

        new_zarr["cell_groups"][index].create_dataset("indices", data=indices)
        new_zarr["cell_groups"][index].create_dataset("indptr", data=indptr)
        mappings.append(mapping_dict)

    new_zarr["cell_groups"].attrs.update(
        {
            "major_version": 1,
            "minor_version": 0,
            "number_groupings": len(columns),
            "grouping_names": columns,
            "group_names": [
                [k for k, v in sorted(mapping_dict.items(), key=lambda item: item[1])][
                    1:
                ]
                for mapping_dict in mappings
            ],
        }
    )
    new_zarr.store.close()


def draw_umap(adata, column: str = "leiden") -> None:
    """Draw UMAP plots for the given AnnData object.

    Args:
        adata (AnnData): The AnnData object containing the data.
        column (str): The column to color the UMAP plot by.
    """
    sc.pl.umap(adata, color=[column])
    plt.show()

    sc.pl.umap(adata, color=["KRT5", "KRT7"], vmax="p95")
    plt.show()

    sc.pl.umap(adata, color=["ACTA2", "PTPRC"], vmax="p95")
    plt.show()


def get_leiden_umap(adata, draw: bool = False):
    """Perform Leiden clustering and UMAP visualization on the given AnnData object.

    Args:
        adata (AnnData): The AnnData object containing the data.
        draw (bool): Whether to draw the UMAP plots.

    Returns:
        AnnData: The AnnData object with Leiden clustering and UMAP results.
    """
    sc.pp.filter_cells(adata, min_genes=5)
    sc.pp.filter_genes(adata, min_cells=5)

    gene_names = adata.var_names
    mean_expression_values = adata.X.mean(axis=0)
    gene_mean_expression_df = pd.DataFrame(
        {"gene_name": gene_names, "mean_expression": mean_expression_values}
    )
    top_genes = gene_mean_expression_df.sort_values(
        by="mean_expression", ascending=False
    ).head(30)
    top_gene_names = top_genes["gene_name"].tolist()

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)

    if draw:
        draw_umap(adata, "leiden")

    return adata


def get_median_expression_table(adata, column: str = "leiden") -> pd.DataFrame:
    """Get the median expression table for the given AnnData object.

    Args:
        adata (AnnData): The AnnData object containing the data.
        column (str): The column to group by.

    Returns:
        pd.DataFrame: The median expression table.
    """
    top_genes = [
        "GATA3",
        "ACTA2",
        "KRT7",
        "KRT8",
        "KRT5",
        "AQP1",
        "SERPINA3",
        "PTGDS",
        "CXCR4",
        "SFRP1",
        "ENAH",
        "MYH11",
        "SVIL",
        "KRT14",
        "CD4",
    ]
    top_gene_indices = [adata.var_names.get_loc(gene) for gene in top_genes]

    clusters = adata.obs[column]
    cluster_data = {}

    for cluster in clusters.unique():
        cluster_cells = adata[clusters == cluster].X
        cluster_expression = cluster_cells[:, top_gene_indices]
        gene_medians = [
            pd.Series(cluster_expression[:, gene_idx]).median()
            for gene_idx in range(len(top_gene_indices))
        ]
        cluster_data[f"Cluster_{cluster}"] = gene_medians

    cluster_expression_df = pd.DataFrame(cluster_data, index=top_genes)
    sorted_columns = sorted(
        cluster_expression_df.columns.values, key=lambda x: int(x.split("_")[-1])
    )
    cluster_expression_df = cluster_expression_df[sorted_columns]
    return cluster_expression_df.T.style.background_gradient(cmap="Greens")


def generate_experiment_file(
    template_path: str,
    output_path: str,
    cells_name: str = "seg_cells",
    analysis_name: str = "seg_analysis",
) -> None:
    """Generate the experiment file for Xenium.

    Args:
        template_path (str): The path to the template file.
        output_path (str): The path to the output file.
        cells_name (str): The name of the cells file.
        analysis_name (str): The name of the analysis file.
    """
    import json

    with open(template_path) as f:
        experiment = json.load(f)

    # experiment["images"].pop("morphology_filepath")
    # experiment["images"].pop("morphology_focus_filepath")

    experiment["xenium_explorer_files"][
        "cells_zarr_filepath"
    ] = f"{cells_name}.zarr.zip"
    experiment["xenium_explorer_files"].pop("cell_features_zarr_filepath")
    experiment["xenium_explorer_files"][
        "analysis_zarr_filepath"
    ] = f"{analysis_name}.zarr.zip"

    with open(output_path, "w") as f:
        json.dump(experiment, f, indent=2)




from pqdm.processes import pqdm  # or from pqdm.processes import pqdm for process backend
import os

def _process_one_cell(args):
    seg_cell_id, seg_cell, area_low, area_high, boundary_method, boundary_voxel_size = args

    if len(seg_cell) < 5:
        return None

    cell_convex_hull = _boundary_from_method(
        seg_cell, boundary_method, boundary_voxel_size
    )
    if cell_convex_hull is None or not isinstance(cell_convex_hull, Polygon):
        return None

    if not (area_low <= cell_convex_hull.area <= area_high):
        return None

    # Get original vertices and remove duplicate closing vertex if present
    cell_vertices = list(cell_convex_hull.exterior.coords)
    if cell_vertices[0] == cell_vertices[-1]:
        cell_vertices = cell_vertices[:-1]
    
    n_vertices = len(cell_vertices)
    
    # Sample up to 16 vertices
    if n_vertices > 16:
        # Evenly sample 16 vertices from original set
        indices = np.linspace(0, n_vertices-1, 16, dtype=int)
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
    seg_df: pd.DataFrame,
    source_path: str,
    output_dir: str,
    cells_filename: str = "seg_cells",
    analysis_filename: str = "seg_analysis",
    xenium_filename: str = "seg_experiment.xenium",
    analysis_df: Optional[pd.DataFrame] = None,
    draw: bool = False,
    cell_id_columns: str = "seg_cell_id",
    area_low: float = 10,
    area_high: float = 100,
    n_jobs: int = 1,
    boundary_method: str = "convex_hull",
    boundary_voxel_size: float = 0.0,
    boundaries: Optional["gpd.GeoDataFrame"] = None,
    boundary_id_column: str = "cell_id",
    boundary_type_column: str = "boundary_type",
    boundary_cell_value: str = "cell",
    boundary_nucleus_value: str = "nucleus",
) -> None:
    source_path = Path(source_path)
    storage = Path(output_dir)
    storage.mkdir(parents=True, exist_ok=True)

    grouped_by = seg_df.groupby(cell_id_columns)

    # Build a lightweight iterable of work items (id, slice, thresholds)
    # NOTE: this will still materialize each group slice, but we avoid copying the whole DF per worker.
    work_iter = (
        (
            seg_cell_id,
            seg_cell,
            area_low,
            area_high,
            boundary_method,
            boundary_voxel_size,
        )
        for seg_cell_id, seg_cell in grouped_by
    )

    # Parallel map with threads (good default). Tune n_jobs.
    # n_jobs = min(32, os.cpu_count() or 8)
    results = pqdm(work_iter, _process_one_cell, n_jobs=n_jobs, desc="Cells", exception_behaviour="immediate")

    # Collate results
    cell_id2old_id: Dict[int, Any] = {}
    cell_id: List[int] = []
    polygon_num_vertices: List[List[int]] = []
    polygon_vertices: List[List[Any]] = []

    # We need a stable incremental id — use enumerate over kept results
    kept = [r for r in results if r is not None]
    for cell_incremental_id, r in enumerate(kept):
        uint_cell_id = cell_incremental_id + 1
        cell_id2old_id[uint_cell_id] = r["seg_cell_id"]
        cell_id.append(uint_cell_id)
        polygon_num_vertices.append(r["cell_num_vertices"])
        polygon_vertices.append(r["cell_vertices"])

    # Flatten vertices exactly as before
    cell_polygon_vertices = get_flatten_version(polygon_vertices)

    source_zarr_store = ZipStore(source_path / "cells.zarr.zip", mode="r") # added this line
    existing_store = zarr.open(source_zarr_store, mode="r")
    output_cells_store = ZipStore(storage / f"{cells_filename}.zarr.zip", mode="w")
    try:
        new_store = zarr.open_group(output_cells_store, mode="w", zarr_format=2)
    except TypeError:
        new_store = zarr.open(output_cells_store, mode="w")

    # Create polygon_sets group with the new structure
    polygon_group = new_store.create_group("polygon_sets")

    # Process cell polygons (set 1)
    # cell_polygons = cells["polygon_vertices"][1]  # Cell polygons are at index 1
    cell_num_vertices = polygon_num_vertices # Cell vertex counts

    # Reshape cell polygons to (n_cells, 50) format
    n_cells = cell_polygon_vertices.shape[0]
    cell_vertices_flat = cell_polygon_vertices.reshape(n_cells, -1)[:, :33]  # Take first 50 values

    set1 = polygon_group.create_group("1")
    set1["cell_index"] = np.arange(1, n_cells + 1, dtype=np.uint32)  # 1-based indexing
    set1["method"] = np.ones(n_cells, dtype=np.uint32)  # All method=1
    set1["num_vertices"] = np.array(cell_num_vertices, dtype=np.int32)
    set1["vertices"] = cell_vertices_flat.astype(np.float32)

    new_store.attrs.update(existing_store.attrs)
    new_store.attrs["number_cells"] = n_cells
    new_store.store.close()
    source_zarr_store.close()

    if analysis_df is None:
        analysis_df = pd.DataFrame(
            [cell_id2old_id[i] for i in cell_id], columns=[cell_id_columns]
        )
        analysis_df["default"] = "segger"

    zarr_df = pd.DataFrame(
        [cell_id2old_id[i] for i in cell_id], columns=[cell_id_columns]
    )
    clustering_df = pd.merge(zarr_df, analysis_df, how="left", on=cell_id_columns)
    clusters_names = [col for col in analysis_df.columns if col != cell_id_columns]

    clusters_dict = {
        cluster: {
            label: idx + 1
            for idx, label in enumerate(
                sorted(np.unique(clustering_df[cluster].dropna()))
            )
        }
        for cluster in clusters_names
    }

    output_analysis_store = ZipStore(storage / f"{analysis_filename}.zarr.zip", mode="w")
    try:
        new_zarr = zarr.open_group(output_analysis_store, mode="w", zarr_format=2)
    except TypeError:
        new_zarr = zarr.open(output_analysis_store, mode="w")
    new_zarr.create_group("/cell_groups")
    for i, cluster in enumerate(clusters_names):
        new_zarr["cell_groups"].create_group(str(i))
        group_values = [clusters_dict[cluster].get(x, 0) for x in clustering_df[cluster]]
        indices, indptr = get_indices_indptr(np.array(group_values))
        new_zarr["cell_groups"][str(i)]["indices"] = indices
        new_zarr["cell_groups"][str(i)]["indptr"] = indptr

    new_zarr["cell_groups"].attrs.update(
        {
            "major_version": 1,
            "minor_version": 0,
            "number_groupings": len(clusters_names),
            "grouping_names": clusters_names,
            "group_names": [
                sorted(clusters_dict[cluster], key=clusters_dict[cluster].get)
                for cluster in clusters_names
            ],
        }
    )
    new_zarr.store.close()

    generate_experiment_file(
        template_path=source_path / "experiment.xenium",
        output_path=storage / xenium_filename,
        cells_name=cells_filename,
        analysis_name=analysis_filename,
    )


# from segger.prediction.boundary import *
# import sys
# import time
# import zarr
# from numcodecs import Blosc
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from typing import Dict, List, Any, Optional
# from shapely.geometry import Polygon
# from pqdm.processes import pqdm
# import json
# from zipfile import ZipFile
# from zarr.storage import ZipStore

def generate_boundary(seg_cell):
    """Generate convex hull boundary for a cell"""
    # Your existing implementation
    points = seg_cell[['x_location', 'y_location']].values
    if len(points) < 3:
        return None
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        return Polygon(points[hull.vertices])
    except:
        return None

# def get_flatten_version(polygon_vertices):
#     """Convert list of vertices to flattened array"""
#     max_vertices = 16
#     n_cells = len(polygon_vertices)
#     flattened = np.full((n_cells, max_vertices * 2), np.nan, dtype=np.float32)
    
#     for i, vertices in enumerate(polygon_vertices):
#         coords_flat = []
#         for vertex in vertices:
#             coords_flat.extend(vertex)
#         flattened[i, :len(coords_flat)] = coords_flat
    
#     return flattened

# def get_indices_indptr(labels):
#     """Get indices and indptr for cell groups"""
#     indices = []
#     indptr = [0]
#     for label in np.unique(labels):
#         if label == 0:
#             continue
#         cell_indices = np.where(labels == label)[0]
#         indices.extend(cell_indices)
#         indptr.append(indptr[-1] + len(cell_indices))
#     return np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int32)


# def _process_cell_chunk(args):
#     """Process a chunk of cells and return results as list"""
#     chunk, area_low, area_high = args
#     chunk_results = []
    
#     for seg_cell_id, seg_cell in chunk:
#         if len(seg_cell) < 5:
#             continue

#         cell_convex_hull = generate_boundary(seg_cell)
#         if cell_convex_hull is None or not isinstance(cell_convex_hull, Polygon):
#             continue

#         if not (area_low <= cell_convex_hull.area <= area_high):
#             continue

#         # Get original vertices
#         cell_vertices = list(cell_convex_hull.exterior.coords)
#         if cell_vertices[0] == cell_vertices[-1]:
#             cell_vertices = cell_vertices[:-1]
        
#         n_vertices = len(cell_vertices)
        
#         # Sample up to 16 vertices
#         if n_vertices > 16:
#             indices = np.linspace(0, n_vertices-1, 16, dtype=int)
#             sampled_vertices = [cell_vertices[i] for i in indices]
#         else:
#             sampled_vertices = cell_vertices
        
#         # Pad with first vertex if needed
#         if len(sampled_vertices) < 16:
#             sampled_vertices += [sampled_vertices[0]] * (16 - len(sampled_vertices))
        
#         # Flatten vertices
#         flattened_vertices = []
#         for vertex in sampled_vertices:
#             flattened_vertices.extend(vertex)
        
#         # Pad to 33 values
#         while len(flattened_vertices) < 33:
#             flattened_vertices.append(flattened_vertices[0] if flattened_vertices else 0.0)
        
#         chunk_results.append({
#             "seg_cell_id": seg_cell_id,
#             "cell_area": float(cell_convex_hull.area),
#             "cell_vertices": flattened_vertices[:33],
#             "cell_num_vertices": len(sampled_vertices),
#         })
    
#     return chunk_results

# def seg2explorer_pqdm_chunked(
#     seg_df: pd.DataFrame,
#     source_path: str,
#     output_dir: str,
#     cells_filename: str = "seg_cells",
#     analysis_filename: str = "seg_analysis",
#     xenium_filename: str = "seg_experiment.xenium",
#     analysis_df: Optional[pd.DataFrame] = None,
#     cell_id_columns: str = "seg_cell_id",
#     area_low: float = 10,
#     area_high: float = 100,
#     n_jobs: int = 1,
#     chunk_size: int = 1000
# ) -> None:
#     source_path = Path(source_path)
#     storage = Path(output_dir)
#     storage.mkdir(parents=True, exist_ok=True)

#     grouped_by = seg_df.groupby(cell_id_columns)
#     total_cells = len(grouped_by)

#     # Create chunks for parallel processing
#     chunks = []
#     current_chunk = []
    
#     for i, (seg_cell_id, seg_cell) in enumerate(grouped_by):
#         current_chunk.append((seg_cell_id, seg_cell))
#         if len(current_chunk) >= chunk_size:
#             chunks.append(current_chunk)
#             current_chunk = []
    
#     if current_chunk:
#         chunks.append(current_chunk)

#     # Process chunks in parallel
#     print(f"Processing {len(chunks)} chunks with {total_cells} total cells...")
#     work_iter = [(chunk, area_low, area_high) for chunk in chunks]
    
#     chunk_results = pqdm(
#         work_iter, 
#         _process_cell_chunk, 
#         n_jobs=n_jobs, 
#         desc="Chunks", 
#         exception_behaviour="immediate"
#     )

#     # Flatten results
#     all_results = []
#     for result_chunk in chunk_results:
#         all_results.extend(result_chunk)
    
#     print(f"Successfully processed {len(all_results)} cells")

#     # Build data structures for zarr
#     cell_id2old_id = {}
#     cell_ids = []
#     polygon_num_vertices = []
#     polygon_vertices = []

#     for cell_incremental_id, result in enumerate(all_results):
#         uint_cell_id = cell_incremental_id + 1
#         cell_id2old_id[uint_cell_id] = result["seg_cell_id"]
#         cell_ids.append(uint_cell_id)
#         polygon_num_vertices.append(result["cell_num_vertices"])
#         polygon_vertices.append(result["cell_vertices"])

#     # Convert to numpy arrays
#     n_cells = len(all_results)
#     cell_vertices_array = np.array(polygon_vertices, dtype=np.float32)

#     # Open source store to copy attributes
#     source_zarr_store = ZipStore(source_path / "cells.zarr.zip", mode="r")
#     existing_store = zarr.open(source_zarr_store, mode="r")

#     # Create new store with compression
#     new_store = zarr.open(storage / f"{cells_filename}.zarr.zip", mode='w')
#     polygon_group = new_store.create_group("polygon_sets")
#     set1 = polygon_group.create_group("1")

#     # Create datasets with compression
#     compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
    
#     set1.create_dataset(
#         'cell_index', 
#         data=np.arange(1, n_cells + 1, dtype=np.uint32),
#         compressor=compressor
#     )
    
#     set1.create_dataset(
#         'method', 
#         data=np.ones(n_cells, dtype=np.uint32),
#         compressor=compressor
#     )
    
#     set1.create_dataset(
#         'num_vertices', 
#         data=np.array(polygon_num_vertices, dtype=np.int32),
#         compressor=compressor
#     )
    
#     set1.create_dataset(
#         'vertices', 
#         data=cell_vertices_array,
#         compressor=compressor
#     )

#     # Copy attributes from source
#     new_store.attrs.update(existing_store.attrs)
#     new_store.attrs["number_cells"] = n_cells
    
#     # Close stores
#     new_store.store.close()
#     source_zarr_store.close()

#     # Create analysis data
#     if analysis_df is None:
#         analysis_df = pd.DataFrame(
#             list(cell_id2old_id.values()), 
#             columns=[cell_id_columns]
#         )
#         analysis_df["default"] = "segger"

#     zarr_df = pd.DataFrame(
#         list(cell_id2old_id.values()), 
#         columns=[cell_id_columns]
#     )
#     clustering_df = pd.merge(zarr_df, analysis_df, how="left", on=cell_id_columns)
#     clusters_names = [col for col in analysis_df.columns if col != cell_id_columns]

#     clusters_dict = {
#         cluster: {
#             label: idx + 1
#             for idx, label in enumerate(
#                 sorted(np.unique(clustering_df[cluster].dropna()))
#             )
#         }
#         for cluster in clusters_names
#     }

#     # Create analysis zarr
#     analysis_store = zarr.open(storage / f"{analysis_filename}.zarr.zip", mode="w")
#     analysis_store.create_group("/cell_groups")
    
#     for i, cluster in enumerate(clusters_names):
#         analysis_store["cell_groups"].create_group(str(i))
#         group_values = [clusters_dict[cluster].get(x, 0) for x in clustering_df[cluster]]
#         indices, indptr = get_indices_indptr(np.array(group_values))
#         analysis_store["cell_groups"][str(i)]["indices"] = indices
#         analysis_store["cell_groups"][str(i)]["indptr"] = indptr

#     analysis_store["cell_groups"].attrs.update(
#         {
#             "major_version": 1,
#             "minor_version": 0,
#             "number_groupings": len(clusters_names),
#             "grouping_names": clusters_names,
#             "group_names": [
#                 sorted(clusters_dict[cluster], key=clusters_dict[cluster].get)
#                 for cluster in clusters_names
#             ],
#         }
#     )
#     analysis_store.store.close()

#     generate_experiment_file(
#         template_path=source_path / "experiment.xenium",
#         output_path=storage / xenium_filename,
#         cells_name=cells_filename,
#         analysis_name=analysis_filename,
#     )

#     print(f"Successfully processed {n_cells} cells")
#     return n_cells
