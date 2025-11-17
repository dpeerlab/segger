from functools import singledispatch
from typing import Literal
import geopandas as gpd
import pandas as pd
import numpy as np
import cupy as cp
import cuspatial
import shapely
import torch
import cudf

from cuspatial.utils.column_utils import (
    contains_only_polygons,
    contains_only_points,
)

# --- Coordinates Conversion ---
@singledispatch
def points_to_coords(data: any) -> np.ndarray | cp.ndarray:
    """Converts various point data formats into a standard coordinate array.

    This is a generic function that uses `singledispatch` to delegate to a
    registered implementation based on the type of the input `data`.

    Parameters
    ----------
    data : any
        The input point data. Supported types must have a registered
        implementation (e.g., list of shapely Points, NumPy/CuPy arrays).

    Returns
    -------
    np.ndarray | cp.ndarray
        A standardized coordinate array of shape (N, 2), either on the CPU
        (NumPy) or GPU (CuPy).

    Raises
    ------
    NotImplementedError
        If no converter is registered for the specific input data type.
    """
    raise NotImplementedError(
        f"No implementation registered for type {type(data).__name__}"
    )

@points_to_coords.register
def _(data: list):
    if not any(data) or not isinstance(data[0], shapely.Point):
        raise ValueError("Input must be a non-empty list of shapely.Point.")
    return np.array([p.coords[0] for p in data])

@points_to_coords.register
def _(data: np.ndarray | cp.ndarray):
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(
            f"Input array must have shape (N, 2), but got {data.shape}."
        )
    return data

@points_to_coords.register
def _(data: torch.Tensor):
    if data.dim() != 2 or data.shape[1] != 2:
        raise ValueError(
            f"Input tensor must have shape (N, 2), but got {data.shape}."
        )
    if data.device == "cpu":
        return data.numpy()
    else:  # "cuda", zero-copy transfer
        return cp.array(data.cuda())

@points_to_coords.register
def _(data: gpd.GeoSeries):
    if data.geometry.empty or data.geom_type.ne('Point').any():
        raise ValueError(
            f"Input must be a non-empty geopandas.GeoSeries of points."
        )
    coords = data.get_coordinates()
    if coords.shape[1] != 2:
        raise ValueError(
            f"Input must be points in 2 dimensions, but got {data.shape[1]}."
        )
    return coords

@points_to_coords.register
def _(data: cuspatial.GeoSeries):
    if data.empty or not contains_only_points(data):
        raise ValueError(
            f"Input must be a non-empty cuspatial.GeoSeries of points."
        )
    return data.points.xy.to_cupy().reshape(-1, 2)  # inherently 2D

# --- Points API ---
def points_to_geoseries(
    data: any,
    backend: Literal['geopandas', 'cuspatial']
) -> gpd.GeoSeries | cuspatial.GeoSeries:
    """
    Converts various point data formats to a specified GeoSeries backend.

    This is a generic function that dispatches to a registered implementation
    based on the type of the input `data`.

    Parameters
    ----------
    data : any
        The input point data. Supported types include:
        - List of shapely Points
        - NumPy/CuPy arrays or Torch tensors of shape (N, 2)
        - GeoPandas/cuSpatial GeoSeries of 2D points
    backend : Literal['geopandas', 'cuspatial']
        The target backend for the output GeoSeries.

    Returns
    -------
    gpd.GeoSeries | cuspatial.GeoSeries
        The converted GeoSeries object.
        
    Raises
    ------
    NotImplementedError
        If no converter is registered for the input data type.
    ValueError
        If the input data has an incorrect shape or format.
    TypeError
        If the backend is not supported.
    """
    if backend not in ['geopandas', 'cuspatial']:
        raise TypeError(
            f"Unsupported backend '{backend}'. Supported backends are "
            f"'geopandas' and 'cuspatial'."
        )
    # Passthrough
    if (backend == 'geopandas' and isinstance(data, gpd.GeoSeries)) or \
       (backend == 'cuspatial' and isinstance(data, cuspatial.GeoSeries)):
        return data
    # Collect points coordinates
    coords = points_to_coords(data)

    # Convert to backend
    if backend == 'geopandas':
        coords = cp.asnumpy(coords)
        points = gpd.GeoSeries(gpd.points_from_xy(*coords.T))
        if isinstance(data, cuspatial.GeoSeries):
            points.index = pd.Index(data.index.values.get())
    else:  # cuspatial
        coords = cp.asarray(coords).ravel().astype('double')
        points = cuspatial.GeoSeries.from_points_xy(coords)
        if isinstance(data, gpd.GeoSeries):
            points.index = cudf.Index(data.index)
    return points


# --- Parts (Vertices and Offsets) Conversion ---
@singledispatch
def polygons_to_parts(data: any) -> tuple[np.ndarray] | tuple[cp.ndarray]:
    """Converts various polygon data into standard vertex and offset arrays.

    This is a generic function that uses `singledispatch` to delegate to a
    registered implementation based on the type of the input `data`.

    Parameters
    ----------
    data : any
        The input polygon data. Supported types must have a registered
        implementation (e.g., list of shapely Polygons, GeoSeries).

    Returns
    -------
    tuple[np.ndarray] | tuple[cp.ndarray]
        A tuple of standardized coordinate arrays of shape (N, 2) for vertices
        and shape (N,) for ring offsets, on the CPU (NumPy) or GPU (CuPy).

    Raises
    ------
    NotImplementedError
        If no converter is registered for the specific input data type.
    """
    raise NotImplementedError(
        f"No implementation registered for type {type(data).__name__}"
    )

@polygons_to_parts.register
def _(data: list):
    if not data or not isinstance(data[0], shapely.Polygon):
        raise ValueError("Input must be a non-empty list of shapely.Polygon.")
    coords = [np.array(p.exterior.coords) for p in data]
    vertices = np.vstack(coords)
    ring_offsets = np.cumsum([0] + [len(c) for c in coords])
    return vertices, ring_offsets

@polygons_to_parts.register
def _(data: torch.Tensor):
    if not data.is_nested or data.layout != torch.jagged:
        raise ValueError(
            "Input tensor must be nested and have 'jagged' layout."
        )
    if data.dim() != 3 or data.shape[-1] != 2:
        raise ValueError(
            "Input tensor must be of shape (N, j2, 2), but got {data.shape}."
        )
    if data.device == "cpu":
        vertices = data.values().numpy()
        ring_offsets = data.offsets().numpy()
        return vertices, ring_offsets
    else:  # "cuda", zero-copy transfer
        vertices = cp.array(data.values().cuda())
        ring_offsets = cp.array(data.offsets().cuda())
        return vertices, ring_offsets

@polygons_to_parts.register
def _(data: gpd.GeoSeries):
    if data.geometry.empty or data.geom_type.ne('Polygon').any():
        raise ValueError(
            f"Input must be a non-empty geopandas.GeoSeries of polygons."
        )
    coords = data.get_coordinates()
    vertices = coords[['x', 'y']].to_numpy()
    _, idx = np.unique(coords.index.to_numpy(), return_index=True)
    ring_offsets = np.sort(np.append(idx, len(coords)))
    return vertices, ring_offsets

@polygons_to_parts.register
def _(data: cuspatial.GeoSeries):
    if data.empty or not contains_only_polygons(data):
        raise ValueError(
            f"Input must be a non-empty cuspatial.GeoSeries of polygons."
        )
    vertices = data.polygons.xy.to_cupy().reshape(-1, 2)
    ring_offsets = data.polygons.ring_offset
    return vertices, ring_offsets

# --- Polygons API ---
def polygons_to_geoseries(
    data: any,
    backend: Literal['geopandas', 'cuspatial']
) -> gpd.GeoSeries | cuspatial.GeoSeries:
    """
    Converts various polygon data formats to a specified GeoSeries backend. 
    
    Polygon geometries must contains exterior rings only; they cannot have 
    interior holes, etc.

    Parameters
    ----------
    data : any
        The input polygon data. Supported types include:
        - List of shapely Polygons
        - Jagged Torch tensors of shape (N, V, 2) where V is the number of 
          vertices per polygon.
        - GeoPandas/cuSpatial GeoSeries of polygons
    backend : Literal['geopandas', 'cuspatial']
        The target backend for the output GeoSeries.

    Returns
    -------
    gpd.GeoSeries | cuspatial.GeoSeries
        The converted GeoSeries object.
        
    Raises
    ------
    NotImplementedError
        If no converter is registered for the input data type.
    ValueError
        If the input data has an incorrect shape or format.
    TypeError
        If the backend is not supported.
    """
    if backend not in ['geopandas', 'cuspatial']:
        raise TypeError(
            f"Unsupported backend '{backend}'. Supported backends are "
            f"'geopandas' and 'cuspatial'."
        )
    # Passthrough
    if (backend == 'geopandas' and isinstance(data, gpd.GeoSeries)) or \
       (backend == 'cuspatial' and isinstance(data, cuspatial.GeoSeries)):
        return data
    # Collect points coordinates
    vertices, ring_offsets = polygons_to_parts(data)

    # Convert to backend
    if backend == 'geopandas':
        vertices = cp.asnumpy(vertices)
        ring_offsets = cp.asnumpy(ring_offsets)
        part_offsets = np.arange(len(ring_offsets), dtype='int32')
        polygons = gpd.GeoSeries(shapely.from_ragged_array(
            shapely.GeometryType.POLYGON,
            vertices,
            (ring_offsets, part_offsets),
        ))
        if isinstance(data, cuspatial.GeoSeries):
            polygons.index = pd.Index(data.index.values.get())
    else:  # cuspatial
        vertices = cp.asarray(vertices).ravel().astype('double')
        ring_offsets = cp.asarray(ring_offsets)
        part_offsets = cp.arange(len(ring_offsets), dtype='int32')
        polygons = cuspatial.GeoSeries.from_polygons_xy(
            vertices,
            ring_offsets,
            part_offsets,
            part_offsets,
        )
        if isinstance(data, gpd.GeoSeries):
            polygons.index = cudf.Index(data.index)
    return polygons

def polygons_to_nested_tensor(
    data: any,
    device: str | None = None,
) -> torch.Tensor:
    """Converts polygon geometries into a nested tensor in jagged layout

    The jagged tensor format is used here for representing polygons with 
    varying numbers of vertices without requiring padding.

    Parameters
    ----------
    data : any
        Input data representing polygon geometries.
    device : str or None, optional
        The device to place the final tensor on ('cuda' or 'cpu'). If
        'cuda', uses `cuspatial` for acceleration. Defaults to None (cpu).

    Returns
    -------
    torch.Tensor
        A nested tensor of shape (N, *, 2), where N is the number of
        polygons and * represents the variable number of vertices for each
        polygon.
    """
    # Convert to universal format (GeoSeries)
    backend = 'cuspatial' if device == 'cuda' else 'geopandas'
    polygons = polygons_to_geoseries(data, backend=backend)
    
    # Build nested tensor from coordinates
    coords = polygons.geometry.get_coordinates()
    _, counts = torch.unique(
        torch.tensor(coords.index.values),
        return_counts=True,
    )
    indices = torch.cumsum(counts, 0)[:-1]
    splits = torch.tensor_split(torch.tensor(coords.values), indices)
    return torch.nested.nested_tensor(
        splits,
        layout=torch.jagged,
        device=device
    )
