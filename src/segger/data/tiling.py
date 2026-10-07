from functools import cached_property
from abc import ABC, abstractmethod
from numpy.typing import ArrayLike
from shapely import box
import geopandas as gpd
import numpy as np
import cupy as cp
import torch
import warnings
import logging

logger = logging.getLogger(__name__)

from .csr import index_to_ptr
from ..geometry import (
    assign_points_to_polygons,
    bounds_to_cuspa,
    bounds_to_geoseries,
    quadtree_leaf_bounds,
)
from ..geometry.quadtree import MAX_QUADTREE_POINTS


class Tiling(ABC):
    """
    An abstract base class for spatial tilings.

    Implementing classes must define the `tiles` property, which returns a
    geopandas GeoSeries. This property should be computed once and cached.
    The base class provides methods to index and mask positions based on tiles.
    """

    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def tiles(self) -> gpd.GeoSeries:
        """
        A collection of Polygon geometries representing the tiles.

        This is an abstract property that must be implemented by subclasses.
        It is recommended to use @cached_property in the implementation for
        on-demand, single-computation generation of tiles.
        """
        ...

    def _check_tiles(self):
        """
        Explicitly ensure `self.tiles` is a collection of Polygon geometries,
        e.g., not MultiPolygon or Line.
        """
        assert self.tiles.geom_type.eq('Polygon').all()
    
    @cached_property
    def bounds(self) -> np.ndarray:
        """Tile bounds as a (K, 4) array of (min_x, min_y, max_x, max_y)."""
        return self.tiles.bounds.to_numpy().astype(np.float64)

    def neighbors(self, margin: float) -> tuple[torch.Tensor, torch.Tensor]:
        """CSR `(ptr, tile_ids)` of the tiles each margined tile can reach.

        1. Grow tiles
        2. Find intersecting (neighboring) tiles
        3. Build CSR representation of neighbors
        """

        # grow tiles without rounding corners
        grown = self.tiles.buffer(margin, join_style='mitre')

        # find overlaps
        hits = gpd.sjoin(
            gpd.GeoDataFrame(geometry=grown),
            gpd.GeoDataFrame(geometry=self.tiles),
            predicate='intersects',
        )

        # get indices
        tile = hits.index.to_numpy()
        neighbor = hits['index_right'].to_numpy()

        # group by tile, listing each tile before its neighbors
        order = np.lexsort((neighbor != tile, tile))
        tile, neighbor = tile[order], neighbor[order]
        ptr, _ = index_to_ptr(
            torch.from_numpy(tile), is_sorted=True, n_buckets=len(grown)
        )

        logger.debug(
            f"Tile neighbors (margin={margin}): {len(grown)} tiles, "
            f"{len(neighbor) / len(grown):.1f} tiles gathered per tile on average"
        )
        return ptr, torch.from_numpy(neighbor)

    def _tile_bounds(self, margin: float = 0.0) -> np.ndarray:
        """Tile bounds as (K, 4) boxes, shrunk inward by `margin`.

        Tiles are axis-aligned boxes in all current tilings. A margin that
        would shrink a tile to nothing is halved until every tile survives.
        """
        bounds = self.bounds
        eff_margin = margin
        while eff_margin > 0:
            shrunk = bounds + [eff_margin, eff_margin, -eff_margin, -eff_margin]
            lost = (shrunk[:, 2:] <= shrunk[:, :2]).any(axis=1)
            if not lost.any():
                return shrunk
            eff_margin = eff_margin / 2 if eff_margin > 1e-6 else 0.0
            warnings.warn(
                f"Margin ({margin}) is too large, causing {int(lost.sum())} "
                f"tile(s) to disappear; retrying with a reduced margin "
                f"({eff_margin}) so their geometries are not dropped from "
                f"the query."
            )
        return bounds.copy()

    def _tile_polygons(self, margin: float = 0.0):
        """Tiles as cuspa polygons, shrunk inward by `margin`; built once
        per margin and cached."""
        cache = self.__dict__.setdefault('_cuspa_tiles', {})
        if margin not in cache:
            cache[margin] = bounds_to_cuspa(self._tile_bounds(margin))
        return cache[margin]

    def _query_tiles(
        self,
        geometry: torch.Tensor,
        inclusive: bool = True,
        margin: float = 0.0,
    ) -> torch.Tensor:
        """Finds which tile contains each point, with an optional margin.

        Parameters
        ----------
        geometry : torch.Tensor
            Point coordinates, shape (N, 2).
        inclusive : bool, optional
            If True, uses an 'intersects' predicate which includes boundaries.
            If False, uses a 'contains' predicate for strict interior
            matches. Defaults to True.
        margin : float, optional
            A non-negative distance to shrink the tiles inward before
            querying. A margin of 0.0 means the original tiles are used.
            Defaults to 0.0.

        Returns
        -------
        torch.Tensor
            A 1D tensor of shape (N,) with the index of the matching tile,
            or -1 if unmatched.

        Raises
        ------
        ValueError
            If geometry shape is invalid, margin is negative, or margin is
            so large that tiles disappear.
        """
        if geometry.dim() != 2 or geometry.shape[-1] != 2:
            raise ValueError(
                f"Input 'geometry' must be a tensor of points of shape (N, 2), "
                f"but got {geometry.shape}."
            )
        if margin < 0:
            raise ValueError(
                f"The margin must be non-negative, but got {margin}."
            )
        predicate = 'intersects' if inclusive else 'contains'
        labels = assign_points_to_polygons(
            geometry,
            self._tile_polygons(margin),
            predicate,
            check_full=(inclusive and margin == 0),
        )
        # Stay on device: a host round-trip costs 4 GB each way at a
        # billion points.
        return torch.as_tensor(labels, device=geometry.device, dtype=torch.int64)

    def label(self, geometry: torch.Tensor) -> torch.Tensor:
        """Assigns a tile index to each point; -1 if unmatched.

        Parameters
        ----------
        geometry : torch.Tensor
            Point coordinates, shape (N, 2).

        Returns
        -------
        torch.Tensor
            A 1D tensor of tile indices, one per point.
        """
        return self._query_tiles(geometry, inclusive=True)

    def mask(self, geometry: torch.Tensor, margin: float) -> torch.Tensor:
        """Marks points that fall inside the tiles after shrinking by `margin`.

        Parameters
        ----------
        geometry : torch.Tensor
            Point coordinates, shape (N, 2).
        margin : float
            The non-negative distance to shrink the tiles inward.

        Returns
        -------
        torch.Tensor
            A 1D boolean tensor, True where the point is inside a shrunk tile.
        """
        labels = self._query_tiles(geometry, inclusive=False, margin=margin)
        return labels != -1

class QuadTreeTiling(Tiling):
    """A tiling system based on a quadtree decomposition of input points.

    This class partitions a 2D space by generating quadtree tiles that
    adapt to the density of the provided positions, ensuring no single
    tile contains more than a specified maximum number of points.

    Parameters
    ----------
    positions : torch.Tensor
        A 2D tensor of coordinates with shape (N, 2) used to generate
        the quadtree.
    max_tile_size : int
        The maximum number of points allowed in any single quadtree tile.
    max_quadtree_points : int, optional
        Subsample positions to at most this many points for quadtree
        construction; `max_tile_size` is rescaled accordingly.
    """
    def __init__(
        self,
        positions: torch.Tensor,
        max_tile_size: int,
        max_quadtree_points: int = MAX_QUADTREE_POINTS,
    ):
        bounds = quadtree_leaf_bounds(
            positions,
            max_tile_size,
            max_points=max_quadtree_points,
        )
        # fastquadtree can produce leaves with zero points; drop them.
        labels = assign_points_to_polygons(
            positions.cpu(), bounds_to_cuspa(bounds), 'intersects'
        )
        counts = cp.asnumpy(cp.bincount(labels[labels >= 0], minlength=len(bounds)))
        if (counts == 0).any():
            logger.warning(f"Dropping {int((counts == 0).sum())} empty quadtree leaf tile(s)")
            bounds = bounds[counts > 0]
        self._tiles = bounds_to_geoseries(bounds)
        self._tile_polygons()

    @property
    def tiles(self) -> gpd.GeoSeries:
        """
        A collection of Polygon geometries representing the boundaries of the
        leaves of the generated QuadTree.
        """
        return self._tiles


### Benchmarking Class ###

class SquareTiling(Tiling):
    """A tiling system based on a uniform square grid.

    This class partitions a 2D space into square tiles of a fixed size,
    covering the full extent of the input positions. Tiles at the boundaries
    may be smaller if the spatial extent is not evenly divisible by the
    side length.

    Parameters
    ----------
    positions : torch.Tensor
        A 2D tensor of coordinates with shape (N, 2) used to determine
        the spatial extent of the tiling.
    side_length : float
        The side length of each square tile. Must be positive.
    """
    def __init__(
        self,
        positions: torch.Tensor,
        side_length: float,
    ):
        if side_length <= 0:
            raise ValueError(
                f"side_length must be positive, but got {side_length}."
            )
        if positions.dim() != 2 or positions.shape[-1] != 2:
            raise ValueError(
                f"positions must be a tensor of shape (N, 2), "
                f"but got {positions.shape}."
            )
        if len(positions) == 0:
            raise ValueError("positions cannot be empty.")
        
        # Store only the spatial extent, not the positions
        self.min_x = positions[:, 0].min().item()
        self.max_x = positions[:, 0].max().item()
        self.min_y = positions[:, 1].min().item()
        self.max_y = positions[:, 1].max().item()
        self.side_length = side_length
        super().__init__()

    @cached_property
    def tiles(self) -> gpd.GeoSeries:
        """
        A collection of Polygon geometries representing square tiles
        covering the spatial extent of the input positions.
        
        Returns
        -------
        gpd.GeoSeries
            A GeoSeries of square Polygon tiles.
        """
        x, y = np.meshgrid(
            np.arange(self.min_x, self.max_x, self.side_length),
            np.arange(self.min_y, self.max_y, self.side_length),
            indexing='ij'
        )
        coords = np.column_stack([
            x.ravel(), y.ravel(),
            np.minimum(x.ravel() + self.side_length, self.max_x),
            np.minimum(y.ravel() + self.side_length, self.max_y)
        ])
        return gpd.GeoSeries([box(*c) for c in coords])
