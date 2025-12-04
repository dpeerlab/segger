import geopandas as gpd
import pandas as pd

def get_polygon_props(
    polygons: gpd.GeoSeries,
    area: bool = True,
    convexity: bool = True,
    elongation: bool = True,
    circularity: bool = True,
) -> pd.DataFrame:
    """
    Computes geometric properties of polygons.

    Parameters
    ----------
    polygons : gpd.GeoSeries
        A GeoSeries containing polygon geometries.
    area : bool, optional
        If True, compute the area of each polygon (default is True).
    convexity : bool, optional
        If True, compute the convexity of each polygon (default is True).
    elongation : bool, optional
        If True, compute the elongation of each polygon (default is True).
    circularity : bool, optional
        If True, compute the circularity of each polygon (default is True).

    Returns
    -------
    props : pd.DataFrame
        A DataFrame containing the computed properties for each polygon.
    """
    props = pd.DataFrame(index=polygons.index, dtype=float)
    if area:
        props["area"] = polygons.area
    if convexity:
        props["convexity"] = polygons.convex_hull.area / polygons.area
    if elongation:
        rects = polygons.minimum_rotated_rectangle()
        props["elongation"] = rects.area / polygons.envelope.area
    if circularity:
        r = polygons.minimum_bounding_radius()
        props["circularity"] = polygons.area / r**2
    return props