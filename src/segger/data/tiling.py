from functools import cached_property
from abc import ABC, abstractmethod
from numpy.typing import ArrayLike
from shapely import box
import geopandas as gpd
import numpy as np
import torch
import cudf

from ..geometry import *


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
    
    def _query_tiles(
        self,
        geometry: torch.Tensor,
        inclusive: bool = True,
        margin: float = 0.0,
    ) -> cudf.DataFrame:
        """Finds which tile contains each geometry, with optional margins.

        This is the core private method for all spatial queries. It handles
        input validation, optional negative buffering (margins), and
        dispatches to the correct spatial join function based on the
        input geometry's shape.

        Parameters
        ----------
        geometry : torch.Tensor
            A tensor of points (shape: N, 2) or polygons (shape: N, V, 2)
            to query against the tiles.
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
            A 1D tensor of shape (N,) where each element is the integer
            index of the first matching tile. Unmatched geometries are
            assigned a label of -1.

        Raises
        ------
        ValueError
            If geometry shape is invalid, margin is negative, or margin is
            so large that tiles disappear.
        """
        # Check inputs
        if geometry.dim() not in [2, 3] or geometry.shape[-1] != 2:
            raise ValueError(
                f"Input 'geometry' must be a tensor of points of shape (N, 2) "
                f"or polygons of shape (N, V, 2), but got {geometry.shape}."
            )
        if margin < 0:
            raise ValueError(
                f"The margin must be non-negative, but got {margin}."
            )
        # Buffer tiles
        tiles = self.tiles
        if margin > 0:
            buffered = tiles.buffer(
                -margin,
                cap_style='square',
                join_style='mitre',
                mitre_limit=margin / 2,
            )
            missing = buffered.is_empty.sum()
            if missing != 0:
                raise ValueError(
                    f"Margin ({margin}) is too large, causing {missing} "
                    f"tile(s) to disappear. Consider using a smaller margin."
                )
            tiles = buffered

        # Spatial query
        predicate = 'intersects' if inclusive else 'contains'
        if geometry.dim() == 2: # points
            result = points_in_polygons(geometry, tiles, predicate)
        else: # polygons
            result = polygons_in_polygons(geometry, tiles, predicate)
        result = result.drop_duplicates('index_query')

        # Format to tensor of indices (-1 where no match found)
        kwargs = dict(device=geometry.device, dtype=torch.int64)
        labels = torch.full((len(geometry),), -1, **kwargs)
        return labels.scatter_(
            dim=0,
            index=torch.tensor(result['index_query'], **kwargs),
            src=  torch.tensor(result['index_match'], **kwargs),
        )

    def label(
        self,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        """Assigns a tile index to each input geometry.

        For each input geometry, this method finds the index of the tile that
        contains it. Geometries on the boundary of a tile are not considered
        a match.

        Parameters
        ----------
        geometry : torch.Tensor
            A tensor of points (shape: N, 2) or polygons (shape: N, V, 2)
            to label.

        Returns
        -------
        torch.Tensor
            A 1D tensor of tile indices corresponding to each input geometry.
            Unmatched geometries are labeled -1.
        """
        return self._query_tiles(geometry, inclusive=True)

    def mask(
        self,
        geometry: torch.Tensor,
        margin: float,
    ) -> torch.Tensor:
        """Creates a boolean mask for geometries within a tile's margin.

        This method identifies which input geometries fall strictly inside
        the tiles after they have been shrunk by the specified `margin`.

        Parameters
        ----------
        geometry : torch.Tensor
            A tensor of points (shape: N, 2) or polygons (shape: N, V, 2)
            to mask.
        margin : float
            The non-negative distance to shrink the tiles inward.

        Returns
        -------
        torch.Tensor
            A 1D boolean tensor where `True` indicates a geometry is
            inside a buffered tile.
        """
        # Spatial query
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
    """
    def __init__(
        self,
        positions: torch.Tensor,
        max_tile_size: int,
    ):
        # Calculate QuadTree on points and set as tiles
        points = points_to_geoseries(positions, backend='cuspatial')
        _, quadtree = get_quadtree_index(
            points,
            max_tile_size,
            with_bounds=True,
        )
        self._tiles = quadtree_to_geoseries(quadtree, backend='geopandas')

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
