"""Toy Xenium dataset for testing.

This module provides functions to load or create a minimal Xenium dataset
for testing purposes. The dataset contains ~1,000 transcripts and ~50 cells,
small enough for fast unit tests but complete enough to test the full pipeline.

When remote data is available, it's downloaded from GitHub Releases.
Otherwise, synthetic data matching the Xenium format is generated.

Usage
-----
>>> # Load all components
>>> transcripts, cells, boundaries = load_toy_xenium()

>>> # Load individual components
>>> transcripts = load_toy_xenium_transcripts()
>>> boundaries = load_toy_xenium_boundaries()

>>> # Create synthetic data (no download required)
>>> transcripts, cells, boundaries = create_synthetic_xenium(
...     n_cells=50,
...     transcripts_per_cell=20,
... )
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import polars as pl

if TYPE_CHECKING:
    import geopandas as gpd


def _try_fetch_or_create(fname: str, create_fn, **kwargs):
    """Try to fetch from registry, fall back to synthetic creation.

    Parameters
    ----------
    fname
        Filename to fetch.
    create_fn
        Function to create synthetic data if fetch fails.
    **kwargs
        Arguments passed to create_fn.

    Returns
    -------
    The loaded or created data.
    """
    try:
        from ._registry import SEGGER_DATA

        path = SEGGER_DATA.fetch(fname, progressbar=True)
        return path
    except Exception as e:
        # Registry not available or file not uploaded yet
        warnings.warn(
            f"Could not fetch {fname} from remote: {e}. "
            "Using synthetic data instead.",
            UserWarning,
            stacklevel=3,
        )
        return None


def load_toy_xenium(
    use_remote: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, "gpd.GeoDataFrame"]:
    """Load the complete toy Xenium dataset.

    Parameters
    ----------
    use_remote
        If True, try to download from remote. If False or download fails,
        create synthetic data.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, gpd.GeoDataFrame]
        (transcripts, cells, boundaries) where:
        - transcripts: DataFrame with transcript positions and QV scores
        - cells: DataFrame with cell metadata
        - boundaries: GeoDataFrame with cell boundary polygons
    """
    transcripts = load_toy_xenium_transcripts(use_remote=use_remote)
    cells = load_toy_xenium_cells(use_remote=use_remote)
    boundaries = load_toy_xenium_boundaries(use_remote=use_remote)

    return transcripts, cells, boundaries


def load_toy_xenium_transcripts(use_remote: bool = True) -> pl.DataFrame:
    """Load toy Xenium transcripts for testing.

    The transcripts DataFrame contains columns matching the Xenium format:
    - x_location, y_location, z_location: Coordinates (microns)
    - feature_name: Gene symbol
    - qv: Phred-scaled quality value (0-40)
    - cell_id: Assigned cell ID from vendor segmentation
    - overlaps_nucleus: 1 if transcript overlaps nucleus

    Parameters
    ----------
    use_remote
        If True, try to download from remote. Falls back to synthetic data.

    Returns
    -------
    pl.DataFrame
        Transcript data with ~1,000 transcripts.
    """
    if use_remote:
        path = _try_fetch_or_create("toy_xenium_transcripts.parquet", None)
        if path is not None:
            return pl.read_parquet(path)

    # Create synthetic data
    data = create_synthetic_xenium(n_cells=50, transcripts_per_cell=20, seed=42)
    return data[0]


def load_toy_xenium_cells(use_remote: bool = True) -> pl.DataFrame:
    """Load toy Xenium cell metadata for testing.

    Parameters
    ----------
    use_remote
        If True, try to download from remote.

    Returns
    -------
    pl.DataFrame
        Cell metadata with ~50 cells.
    """
    if use_remote:
        path = _try_fetch_or_create("toy_xenium_cells.parquet", None)
        if path is not None:
            return pl.read_parquet(path)

    # Create synthetic data
    data = create_synthetic_xenium(n_cells=50, transcripts_per_cell=20, seed=42)
    return data[1]


def load_toy_xenium_boundaries(use_remote: bool = True) -> "gpd.GeoDataFrame":
    """Load toy Xenium cell boundaries for testing.

    Parameters
    ----------
    use_remote
        If True, try to download from remote.

    Returns
    -------
    gpd.GeoDataFrame
        Cell boundary polygons.
    """
    import geopandas as gpd

    if use_remote:
        path = _try_fetch_or_create("toy_xenium_boundaries.parquet", None)
        if path is not None:
            return gpd.read_parquet(path)

    # Create synthetic data
    data = create_synthetic_xenium(n_cells=50, transcripts_per_cell=20, seed=42)
    return data[2]


def create_synthetic_xenium(
    n_cells: int = 50,
    transcripts_per_cell: int = 20,
    n_genes: int = 100,
    image_size: float = 100.0,
    cell_radius: float = 5.0,
    seed: Optional[int] = None,
) -> tuple[pl.DataFrame, pl.DataFrame, "gpd.GeoDataFrame"]:
    """Create synthetic Xenium-like data for testing.

    Generates realistic-looking spatial transcriptomics data matching
    the 10x Genomics Xenium format, useful for unit tests that don't
    require real data.

    Parameters
    ----------
    n_cells
        Number of cells to generate.
    transcripts_per_cell
        Average number of transcripts per cell.
    n_genes
        Number of unique genes in the panel.
    image_size
        Size of the spatial region (microns).
    cell_radius
        Average cell radius (microns).
    seed
        Random seed for reproducibility.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, gpd.GeoDataFrame]
        (transcripts, cells, boundaries)

    Examples
    --------
    >>> transcripts, cells, boundaries = create_synthetic_xenium(
    ...     n_cells=50,
    ...     transcripts_per_cell=20,
    ...     seed=42,
    ... )
    >>> print(len(transcripts))
    1000
    >>> print(len(boundaries))
    50
    """
    import geopandas as gpd
    from shapely.geometry import Point, Polygon

    if seed is not None:
        np.random.seed(seed)

    # Generate cell centers (ensure no overlap)
    cell_centers = _generate_non_overlapping_points(
        n_points=n_cells,
        min_dist=cell_radius * 2.5,
        image_size=image_size,
        seed=seed,
    )

    # Generate gene names
    gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
    # Add some control probes for quality filter testing
    control_probes = [
        "NegControlProbe_0001",
        "NegControlProbe_0002",
        "antisense_Gene_0001",
        "BLANK_0001",
    ]
    all_genes = gene_names + control_probes

    # Generate transcripts
    transcripts_data = []
    transcript_id = 0

    for cell_idx, (cx, cy) in enumerate(cell_centers):
        cell_id = f"cell_{cell_idx:04d}"
        n_tx = np.random.poisson(transcripts_per_cell)
        n_tx = max(5, n_tx)  # At least 5 transcripts per cell

        for _ in range(n_tx):
            # Random position within cell
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, cell_radius * 0.9)
            x = cx + r * np.cos(angle)
            y = cy + r * np.sin(angle)
            z = np.random.uniform(0, 10)  # z-stack position

            # Random gene (mostly real genes, some control probes)
            if np.random.random() < 0.05:
                gene = np.random.choice(control_probes)
            else:
                gene = np.random.choice(gene_names)

            # QV score (mostly high quality, some low)
            if np.random.random() < 0.9:
                qv = np.random.uniform(20, 40)  # High quality
            else:
                qv = np.random.uniform(5, 20)  # Low quality

            # Overlaps nucleus (transcripts near center)
            overlaps_nucleus = 1 if r < cell_radius * 0.5 else 0

            transcripts_data.append({
                "transcript_id": f"tx_{transcript_id:08d}",
                "x_location": x,
                "y_location": y,
                "z_location": z,
                "feature_name": gene,
                "qv": qv,
                "cell_id": cell_id,
                "overlaps_nucleus": overlaps_nucleus,
            })
            transcript_id += 1

    # Add some unassigned transcripts (outside cells)
    n_unassigned = int(n_cells * transcripts_per_cell * 0.1)
    for _ in range(n_unassigned):
        x = np.random.uniform(0, image_size)
        y = np.random.uniform(0, image_size)
        z = np.random.uniform(0, 10)
        gene = np.random.choice(gene_names)
        qv = np.random.uniform(15, 35)

        transcripts_data.append({
            "transcript_id": f"tx_{transcript_id:08d}",
            "x_location": x,
            "y_location": y,
            "z_location": z,
            "feature_name": gene,
            "qv": qv,
            "cell_id": "UNASSIGNED",
            "overlaps_nucleus": 0,
        })
        transcript_id += 1

    transcripts = pl.DataFrame(transcripts_data)

    # Generate cell metadata
    cells_data = []
    for cell_idx, (cx, cy) in enumerate(cell_centers):
        cell_id = f"cell_{cell_idx:04d}"
        n_tx = len([t for t in transcripts_data if t["cell_id"] == cell_id])

        cells_data.append({
            "cell_id": cell_id,
            "x_centroid": cx,
            "y_centroid": cy,
            "transcript_counts": n_tx,
            "cell_area": np.pi * cell_radius**2,
        })

    cells = pl.DataFrame(cells_data)

    # Generate cell boundaries (circular polygons)
    boundaries_data = []
    geometries = []

    for cell_idx, (cx, cy) in enumerate(cell_centers):
        cell_id = f"cell_{cell_idx:04d}"

        # Create circular polygon with some noise
        n_vertices = 32
        angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
        radii = cell_radius * (1 + np.random.uniform(-0.1, 0.1, n_vertices))
        vertices = [
            (cx + r * np.cos(a), cy + r * np.sin(a))
            for r, a in zip(radii, angles)
        ]
        vertices.append(vertices[0])  # Close polygon

        polygon = Polygon(vertices)
        geometries.append(polygon)
        boundaries_data.append({"cell_id": cell_id})

    boundaries = gpd.GeoDataFrame(
        boundaries_data,
        geometry=geometries,
    )

    return transcripts, cells, boundaries


def _generate_non_overlapping_points(
    n_points: int,
    min_dist: float,
    image_size: float,
    seed: Optional[int] = None,
    max_attempts: int = 1000,
) -> list[tuple[float, float]]:
    """Generate non-overlapping point positions.

    Uses rejection sampling to place points with minimum distance constraint.

    Parameters
    ----------
    n_points
        Number of points to generate.
    min_dist
        Minimum distance between points.
    image_size
        Size of the region.
    seed
        Random seed.
    max_attempts
        Maximum attempts per point before reducing constraints.

    Returns
    -------
    list[tuple[float, float]]
        List of (x, y) coordinates.
    """
    if seed is not None:
        np.random.seed(seed)

    points = []
    margin = min_dist / 2

    for _ in range(n_points):
        for attempt in range(max_attempts):
            x = np.random.uniform(margin, image_size - margin)
            y = np.random.uniform(margin, image_size - margin)

            # Check distance to existing points
            valid = True
            for px, py in points:
                dist = np.sqrt((x - px)**2 + (y - py)**2)
                if dist < min_dist:
                    valid = False
                    break

            if valid:
                points.append((x, y))
                break
        else:
            # Failed to place point, add anyway (for small regions)
            x = np.random.uniform(margin, image_size - margin)
            y = np.random.uniform(margin, image_size - margin)
            points.append((x, y))

    return points


def save_toy_xenium(
    output_dir: Path | str,
    n_cells: int = 50,
    transcripts_per_cell: int = 20,
    seed: int = 42,
) -> dict[str, Path]:
    """Save synthetic Xenium data to files.

    Useful for creating test fixtures or uploading to GitHub Releases.

    Parameters
    ----------
    output_dir
        Directory to save files.
    n_cells
        Number of cells.
    transcripts_per_cell
        Transcripts per cell.
    seed
        Random seed.

    Returns
    -------
    dict[str, Path]
        Dictionary mapping file type to saved path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transcripts, cells, boundaries = create_synthetic_xenium(
        n_cells=n_cells,
        transcripts_per_cell=transcripts_per_cell,
        seed=seed,
    )

    paths = {}

    # Save transcripts
    tx_path = output_dir / "toy_xenium_transcripts.parquet"
    transcripts.write_parquet(tx_path)
    paths["transcripts"] = tx_path

    # Save cells
    cells_path = output_dir / "toy_xenium_cells.parquet"
    cells.write_parquet(cells_path)
    paths["cells"] = cells_path

    # Save boundaries
    bd_path = output_dir / "toy_xenium_boundaries.parquet"
    boundaries.to_parquet(bd_path)
    paths["boundaries"] = bd_path

    return paths
