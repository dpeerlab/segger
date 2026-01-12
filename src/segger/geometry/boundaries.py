from __future__ import annotations

import geopandas as gpd
import polars as pl
import shapely
from shapely.geometry import LineString, MultiPoint, Point


def _points_to_polygon(
    coords: list[list[float]] | "np.ndarray",
    concave_ratio: float,
    min_buffer: float,
):
    """
    Convert a set of 2D points to a polygon hull.
    """
    # Small-cardinality fallbacks
    if len(coords) == 1:
        return Point(coords[0]).buffer(min_buffer)
    if len(coords) == 2:
        return LineString(coords).buffer(min_buffer)

    multipoint = MultiPoint(coords)
    concave = getattr(shapely, "concave_hull", None)

    if concave is not None:
        try:
            geom = concave(multipoint, ratio=concave_ratio)
        except Exception:
            geom = multipoint.convex_hull
    else:
        geom = multipoint.convex_hull

    # Ensure validity
    return geom.buffer(0)


def generate_cell_boundaries(
    transcripts: pl.DataFrame,
    x_column: str,
    y_column: str,
    cell_id_column: str,
    concave_ratio: float = 0.6,
    min_buffer: float = 0.5,
) -> gpd.GeoDataFrame:
    """
    Build polygon boundaries per cell from assigned transcripts.

    Parameters
    ----------
    transcripts : pl.DataFrame
        Transcript-level table containing coordinates and cell assignments.
    x_column, y_column : str
        Column names for x/y coordinates.
    cell_id_column : str
        Column containing the assigned cell identifier.
    concave_ratio : float, default=0.6
        Ratio passed to ``shapely.concave_hull`` when available. Values closer
        to 0 produce tighter hulls; 1 approximates the convex hull.
    min_buffer : float, default=0.5
        Buffer used when only one or two points are available to avoid
        degenerate polygons.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with one row per cell and polygon geometry.
    """
    # Early exit on empty inputs
    if transcripts.is_empty():
        return gpd.GeoDataFrame(
            columns=[cell_id_column, "n_transcripts"], geometry=[]
        )

    subset = transcripts.select([cell_id_column, x_column, y_column]).drop_nulls(
        cell_id_column
    )
    if subset.is_empty():
        return gpd.GeoDataFrame(
            columns=[cell_id_column, "n_transcripts"], geometry=[]
        )

    pdf = subset.to_pandas()
    records: list[dict] = []
    geoms = []

    for cell_id, group in pdf.groupby(cell_id_column):
        coords = group[[x_column, y_column]].to_numpy()
        records.append(
            {cell_id_column: cell_id, "n_transcripts": coords.shape[0]}
        )
        geoms.append(
            _points_to_polygon(
                coords=coords,
                concave_ratio=concave_ratio,
                min_buffer=min_buffer,
            )
        )

    return gpd.GeoDataFrame(records, geometry=geoms)
