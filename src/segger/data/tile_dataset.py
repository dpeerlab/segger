from torch_geometric.loader import DynamicBatchSampler
from torch_geometric.data.storage import NodeStorage
from torch_geometric.data import Data, HeteroData
from torch.utils.data import Dataset
from torch_geometric.utils.map import map_index
from torch_geometric.index import index2ptr
import logging
import shapely
import torch


from .partition import PartitionDataset
from .tiling import Tiling
from .._patches import _chunked_nonzero

logger = logging.getLogger(__name__)


def query_ptr(csr, query) -> torch.Tensor:
    """Gather values for bucket(s) `query` from a `(ptr, values)` CSR.

    `query` may be a scalar (one bucket) or a 1-D tensor (concatenated in
    the given order).
    """
    ptr, values = csr

    # single value
    if not (torch.is_tensor(query) and query.dim() > 0):
        q = int(query)
        return values[ptr[q]:ptr[q + 1]]

    # tensor of values
    starts = ptr[query]
    ends = ptr[query + 1]
    counts = ends - starts
    total = int(counts.sum())
    if total == 0:
        return values.new_empty(0)
    base = torch.repeat_interleave(starts, counts)
    within = (torch.arange(total, device=values.device) - torch.repeat_interleave(counts.cumsum(0) - counts, counts))
    return values[base + within]


class TileFitDataset(PartitionDataset):
    """
    Partitions a PyG graph based on a geometric tiling of its nodes.

    This class extends `PartitionDataset` to create partitions by assigning
    each node to a tile based on its spatial coordinates. It can also add a
    mask for nodes within a certain margin of tile boundaries and optionally
    remove the geometry data after partitioning.

    Parameters
    ----------
    data : Data or HeteroData
        The input graph object containing node geometries.
    tiling : Tiling
        A tiling object with `label` and `mask` methods to partition nodes.
    margin : float
        The margin distance used to create the boolean mask.
    geometry_key : str, optional
        The attribute key for accessing node geometry data, by default
        'geometry'.
    drop_geometry : bool, optional
        If True, removes the geometry attribute from the data after
        partitioning, by default True.
    """
    def __init__(
        self,
        data: Data | HeteroData,
        tiling: Tiling,
        margin: float,
        geometry_key: str = 'geometry',
        clone: bool = True,
        drop_geometry: bool = True,
    ):
        """Initializes and tiles the dataset"""
        self.geometry_key = geometry_key
        self._validate_data(data)

        # Create partition from tiling object and update data
        self.tiling = tiling
        self.margin = margin
        partition = self._get_partition(data)
        
        # Partition graph by tiling
        # Note: self.data and self.partition are set inside super.__init__()
        super().__init__(data=data, partition=partition, clone=clone)
        self.data = self._mask_data(self.data)
        if drop_geometry:
            self.data = self._drop_geometry(self.data)

    def _validate_geometry(
        self,
        node_store: NodeStorage,
        store_name: str,
    ):
        """Checks that 'node_store' has a valid geometry attribute."""
        if self.geometry_key not in node_store.node_attrs():
            raise AttributeError(
                f"{store_name} is missing '{self.geometry_key}' attribute."
            )
        geometry = node_store[self.geometry_key]
        if not isinstance(geometry, torch.Tensor):
            raise TypeError(
                f"The '{self.geometry_key}' attribute for {store_name} must be "
                f"a torch.Tensor, but got {type(geometry).__name__}."
            )
        if geometry.dim() not in [2, 3] or geometry.shape[-1] != 2:
            raise ValueError(
                f"The '{self.geometry_key}' attribute for {store_name} must "
                f"have shape (num_nodes, 2) or (num_nodes, num_vertices, 2), "
                f"but got shape {list(geometry.shape)}."
            )

    def _validate_data(self, data: Data | HeteroData):
        """
        Checks 'data' is a Pytorch Geometric data object, that all node types
        have valid geometry attributes, and that 'mask' does not already exist
        as an attribute.
        """
        if isinstance(data, Data):
            store_name = "The 'data' object"
            self._validate_geometry(data, store_name)
            if 'mask' in data:
                raise KeyError(
                    f"{store_name} must not contain an attribute 'mask'."
                )
        elif isinstance(data, HeteroData):
            if not data.node_types:
                return
            for node_type in data.node_types:
                store_name = f"Node type '{node_type}' in the 'data' object"
                self._validate_geometry(data[node_type], store_name)
                if 'mask' in data[node_type]:
                    raise KeyError(
                        f"{store_name} must not contain an attribute 'mask'."
                    )
        else:
            raise TypeError(
                f"Input must be a PyG Data or HeteroData object, but got "
                f"{type(data).__name__}."
            )

    def _get_partition(self, data: Data | HeteroData) -> torch.Tensor:
        """
        Generates partition labels for all nodes using the tiling object.
        """
        n_tiles = len(self.tiling.tiles)
        if isinstance(data, HeteroData):
            partition = dict()
            for node_type in data.node_types:
                geom = data[node_type][self.geometry_key]
                logger.debug(
                    f"TileFit label '{node_type}': {len(geom)} geoms vs {n_tiles} tiles → quadtree"
                )
                partition[node_type] = self.tiling.label(geom)
            return partition
        else:  # isinstance(data, Data)
            geom = data[self.geometry_key]
            logger.debug(f"TileFit label: {len(geom)} geoms vs {n_tiles} tiles → quadtree")
            return self.tiling.label(geom)

    def _mask_data(self, data: Data | HeteroData) -> Data | HeteroData:
        """
        Adds a boolean 'mask' attribute to each node indicating whether it is
        within a specified margin of a tile's boundary.
        """
        n_tiles = len(self.tiling.tiles)
        if isinstance(data, HeteroData):
            for node_type in data.node_types:
                geom = data[node_type][self.geometry_key]
                logger.debug(
                    f"TileFit mask '{node_type}': {len(geom)} geoms vs {n_tiles} tiles "
                    f"(margin={self.margin}) → quadtree"
                )
                data[node_type]['mask'] = self.tiling.mask(geom, self.margin)
        else:  # isinstance(data, Data)
            geom = data[self.geometry_key]
            logger.debug(
                f"TileFit mask: {len(geom)} geoms vs {n_tiles} tiles "
                f"(margin={self.margin}) → quadtree"
            )
            data['mask'] = self.tiling.mask(geom, self.margin)
        return data

    def _drop_geometry(self, data: Data | HeteroData) -> Data | HeteroData:
        """Removes the geometry attribute from all node stores."""
        if isinstance(data, HeteroData):
            for node_type in data.node_types:
                del data[node_type][self.geometry_key]
        else:  # isinstance(data, Data)
            del data[self.geometry_key]
        return data


class TilePredictDataset(Dataset):
    """A dataset for iterating over spatial tiles with overlapping margins.
    
    This dataset provides subgraphs of a larger graph based on spatial
    tiling. Each item corresponds to a tile, returning the subgraph of
    nodes that fall within the tile boundaries plus a specified margin.
    
    Parameters
    ----------
    data : Data | HeteroData
        The full graph dataset containing node positions and edges.
    tiling : Tiling
        A Tiling object that defines the spatial partitioning.
    margin : float, optional
        The distance to extend tile boundaries for including overlapping
        nodes. Positive values expand tiles outward, negative values
        shrink them inward. Defaults to 0.0.
    """
    def __init__(
        self,
        data: Data | HeteroData,
        tiling: Tiling,
        margin: float = 0.0,
    ):
        """Initializes and partitions the dataset."""
        self.data = data
        self.tiling = tiling
        self.margin = float(margin)
        self._is_hetero = isinstance(self.data, HeteroData)

        # Validate presence of positions.
        if self._is_hetero:
            missing = []
            for node_type in self.data.node_types:
                if 'pos' not in self.data[node_type].node_attrs():
                    missing.append(node_type)
            if missing:
                raise ValueError(
                    f"Missing 'pos' attribute for node type: "
                    f"{', '.join(missing)}"
                )
        elif 'pos' not in self.data.node_attrs():
            raise ValueError("Graph must contain 'pos' attribute.")

        # Precompute CSRs for fast per-tile subsetting (one-time cost).
        if self._is_hetero:
            logger.debug("Building tile/edge pointers for fast subsetting...")
            self._tile_ptr_inner = self._build_tile_ptr(margin=0.0)
            self._tile_ptr_outer = self._build_tile_ptr(margin=self.margin)
            self._edges_ptr = self._build_edge_ptr()

    def _build_tile_ptr(self, margin: float) -> dict:
        """`{node_type: (ptr, node_ids)}` CSR keyed by tile id."""
        n_tiles = len(getattr(self.tiling, 'tiles', self.tiling))
        out = {}
        for nt in self.data.node_types:
            pairs = self._get_tiles_to_nodes_edges(nt, margin=margin)
            out[nt] = (index2ptr(pairs[0], size=n_tiles), pairs[1])
        return out

    def _build_edge_ptr(self) -> dict:
        """`{edge_type: (ptr, edge_positions)}` CSR keyed by src node id.

        Assumes `edge_index` is sorted by src; values are the identity range
        so `query_ptr` returns original-column positions in `edge_index`.
        """
        out = {}
        for et in self.data.edge_types:
            ei = self.data[et].edge_index
            assert (ei[0][1:] >= ei[0][:-1]).all(), f"edge_index[0] for {et} not sorted by src"
            out[et] = (
                index2ptr(ei[0], size=len(self.data[et[0]])),
                torch.arange(ei.shape[1], device=ei.device),
            )
        return out

    def __len__(self) -> int:
        """Number of tiles in the dataset."""
        return len(self.tiling.tiles)

    def __getitem__(self, idx: int) -> Data | HeteroData:
        """Get the graph tile associated at location `index`. 
        
        Initializes an empty Data or HeteroData object and populates with node
        and edge attributes associated with the indexed graph partition. Other
        non-node/edge attributes are populated without subsetting.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Requested {idx}, but tiling only contains {len(self)} tiles."
            )
        geometry = self.tiling.tiles[idx]
        return self._subset(geometry)

    def _get_tiles_to_nodes_edges(self, node_type: str, margin: float) -> torch.Tensor:
        """
        Create edges `(tile_id, node_id)` for nodes in each tile's margined bbox.

        Return tuples, sorted by `tile_id`.
        """
        pos: torch.Tensor = self.data[node_type]['pos'].to(torch.float32)
        tiles_geom = getattr(self.tiling, 'tiles', self.tiling)
        bounds = tiles_geom.bounds.to_numpy().astype("float32")
        bounds = torch.from_numpy(bounds).to(pos.device)
        bounds[:, :2] -= margin
        bounds[:, 2:] += margin

        # Chunk tiles & nodes, cap at ~128 MB intermediate
        K, N = bounds.shape[0], pos.shape[0]
        budget = 2 ** 27  # ~128M bools = 128MB
        chunk_K = max(8, min(256, K))
        chunk_N = max(1, min(N, budget // (8 * max(chunk_K, 1))))

        tile_ids, node_ids = [], []

        # for each batch of tiles
        for s_t in range(0, K, chunk_K):
            ch = bounds[s_t:min(s_t + chunk_K, K)]

            # for each batch of nodes
            for s_n in range(0, N, chunk_N):
                px = pos[s_n:min(s_n + chunk_N, N), 0]
                py = pos[s_n:min(s_n + chunk_N, N), 1]

                # create boundary mask. results in a (chunked) binary matrix of (chunk_k, chunk_n) where "True" indicates assignment
                m = (
                    (ch[:, None, 0] <= px[None, :]) & (ch[:, None, 2] >  px[None, :]) &
                    (ch[:, None, 1] <= py[None, :]) & (ch[:, None, 3] >  py[None, :])
                )

                # extract pairs
                ki, ni = torch.nonzero(m, as_tuple=True)
                tile_ids.append(ki + s_t)
                node_ids.append(ni + s_n)
        
        tile_ids = torch.cat(tile_ids)
        node_ids = torch.cat(node_ids)

        # sort by tile_id (and preserve node order)
        perm = torch.argsort(tile_ids, stable=True)
        return torch.stack([tile_ids[perm], node_ids[perm]], 0)

    def _subset_new(self, idx) -> Data | HeteroData:
        """Subset the Heterograph to nodes and edges within tile `idx`.

        Uses CSRs precomputed in `__init__` (`_tile_ptr_outer`,
        `_tile_ptr_inner`, `_edges_ptr`).
        """
        subset = HeteroData()

        # create nodes
        for node_type in self.data.node_types:

            # get subset
            nodes_subset_idx = query_ptr(self._tile_ptr_outer[node_type], idx)

            # populate metadata
            for key, value in self.data[node_type].items():
                if key == 'num_nodes':
                    subset[node_type].num_nodes = len(nodes_subset_idx)
                elif self.data[node_type].is_node_attr(key):
                    subset[node_type][key] = value[nodes_subset_idx]
                else:
                    subset[node_type][key] = value

            # get mask (mask for nodes within margined tiles)
            nodes_margin_idx = query_ptr(self._tile_ptr_inner[node_type], idx)
            subset[node_type]['predict_mask'] = torch.isin(nodes_subset_idx, nodes_margin_idx)


        # create edges
        for edge_type in self.data.edge_types:

            # get src and dst nodes
            src, _, dst = edge_type
            src_subset = query_ptr(self._tile_ptr_outer[src], idx)
            dst_subset = query_ptr(self._tile_ptr_outer[dst], idx)

            # get edges between src and dst subsets
            edge_src_subset_idx = query_ptr(self._edges_ptr[edge_type], src_subset)
            candidate_edges = self.data[edge_type].edge_index[:, edge_src_subset_idx]
            edge_dst_subset_idx = torch.isin(candidate_edges[1], dst_subset)
            kept_orig = edge_src_subset_idx[edge_dst_subset_idx]
            edge_index_new = candidate_edges[:, edge_dst_subset_idx]

            # map indices to new subset
            src_index, _ = map_index(edge_index_new[0], src_subset, max_index=len(self.data[src]))
            dst_index, _ = map_index(edge_index_new[1], dst_subset, max_index=len(self.data[dst]))
            edge_index_mapped = torch.stack([src_index, dst_index], dim=0)

            # populate heterodata
            for key, value in self.data[edge_type].items():
                if key == 'edge_index':
                    subset[edge_type].edge_index = edge_index_mapped
                elif self.data[edge_type].is_edge_attr(key):
                    subset[edge_type][key] = value[kept_orig]
                else:
                    subset[edge_type][key] = value
        
        return subset




    def _subset(self, bounds: shapely.Polygon) -> Data | HeteroData:
        """Slices all node attributes within bounds.

        TODO: Long Description.
        """
        inner = bounds.bounds
        outer = bounds.buffer(self.margin).bounds
        
        if self._is_hetero:
            subset = dict()
            p_mask = dict()
            for node_type in self.data.node_types:
                pos: torch.Tensor = self.data[node_type]['pos']
                # Row indices of masked elements inside tile w/ margin
                subset[node_type] = _chunked_nonzero(
                    (pos[:, 0] >= outer[0]) &
                    (pos[:, 0] <  outer[2]) &
                    (pos[:, 1] >= outer[1]) &
                    (pos[:, 1] <  outer[3])
                )
                p_mask[node_type] = (
                    (pos[subset[node_type], 0] >= inner[0]) &
                    (pos[subset[node_type], 0] <= inner[2]) &
                    (pos[subset[node_type], 1] >= inner[1]) &
                    (pos[subset[node_type], 1] <= inner[3])
                )
            sample = self.data.subgraph(subset)
            sample.set_value_dict('predict_mask', p_mask)
            return sample

        else:  # is homogenous Data
            pos: torch.Tensor = self.data['pos']
            subset = (
                (pos[:, 0] >= outer[0]) &
                (pos[:, 0] <  outer[2]) &
                (pos[:, 1] >= outer[1]) &
                (pos[:, 1] <  outer[3])
            )
            subset = _chunked_nonzero(subset)
            sample = self.data.subgraph(subset)
            sample['predict_mask'] = (
                (pos[subset, 0] >= inner[0]) &
                (pos[subset, 0] <= inner[2]) &
                (pos[subset, 1] >= inner[1]) &
                (pos[subset, 1] <= inner[3])
            )
            return sample


class DynamicBatchSamplerPatch(DynamicBatchSampler):
    """TODO: Description
    """
    def __len__(self):
        return len(self.dataset)  # ceiling on dataset length
