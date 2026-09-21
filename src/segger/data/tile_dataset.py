from torch_geometric.loader import DynamicBatchSampler
from torch_geometric.data.storage import NodeStorage
from torch_geometric.data import Data, HeteroData
from torch.utils.data import Dataset
import logging
import shapely
import torch


from .csr import index_to_ptr, query_ptr
from .partition import PartitionDataset
from .tiling import Tiling

logger = logging.getLogger(__name__)


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

    Store only tiliing without margins. Instead, add margins from tile
    and its neighbours -> reduces memory load, avoids integer overflows
    and avoids rebuilding this.
    
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
            self._tile_ptr = self._build_tile_ptr()
            # Tile sizes on the host: reading them off the CSR would sync
            # once per item.
            self._tile_sizes = {
                nt: (ptr[1:] - ptr[:-1]).cpu()
                for nt, (ptr, _) in self._tile_ptr.items()
            }
            # Store neighboring tiles
            self._neighbor_ptr = self._build_neighbor_ptr()
            self._margin_bounds = self._build_margin_bounds()
            self._edges_ptr = self._build_edge_ptr()

    @property
    def _device(self) -> torch.device:
        """Device the graph lives on."""
        node_type = self.data.node_types[0]
        return self.data[node_type]['pos'].device

    def _build_tile_ptr(self) -> dict:
        """CSR {node_type: (ptr[tile_id], node_id)} over the unmargined tiles.

        Tiles partition the plane, so each node appears once. Labels come
        from the tiling itself, the same cuspa query the training path uses.
        """
        n_tiles = len(self.tiling.tiles)
        out = {}
        for nt in self.data.node_types:
            pos = self.data[nt]['pos']
            logger.debug(
                f"TilePredict CSR '{nt}': {pos.shape[0]} nodes vs "
                f"{n_tiles} tiles"
            )
            out[nt] = index_to_ptr(self.tiling.label(pos), n_buckets=n_tiles)
        return out

    def _build_neighbor_ptr(self) -> tuple[torch.Tensor, torch.Tensor]:
        """CSR (ptr[tile_id], tile_id) of tiles reachable within the margin."""
        ptr, values = self.tiling.neighbors(self.margin)
        return ptr.to(self._device), values.to(self._device)

    def _build_margin_bounds(self) -> torch.Tensor:
        """Tile bounds grown by the margin, as a (K, 4) tensor."""
        m = self.margin
        bounds = self.tiling.bounds + [-m, -m, m, m]
        return torch.as_tensor(
            bounds, dtype=torch.float32, device=self._device
        )

    def _build_edge_ptr(self) -> dict:
        """Builds CSR-like structure of {edge_type: (ptr[src-node], dst-node)} for edges in each tile.

        Assumes `edge_index` is sorted by src; values are the identity range
        so `query_ptr` returns original-column positions in `edge_index`.
        """
        out = {}
        for et in self.data.edge_types:
            ei = self.data[et].edge_index
            assert (ei[0][1:] >= ei[0][:-1]).all(), f"edge_index[0] for {et} not sorted by src"
            n_src = self.data[et[0]]["pos"].shape[0]
            out[et] = index_to_ptr(ei[0], is_sorted=True, n_buckets=n_src)
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
        return self._subset_new(idx)

    def _nodes_in_margin(
        self,
        node_type: str,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Graph stores nodes within tiles, but not in the extended prediction margin.
        
        - Get transcripts from the node + its neighbors within the margin.
        - Subset nodes within prediction margin

        Note: This could also be a simple cuspa spatial join, but the overhead is much larger than 
        using this pre-computed CSR structure.
        """

        # query the tile and its neighbors, own nodes first
        tiles = query_ptr(self._neighbor_ptr, idx)
        candidates = query_ptr(self._tile_ptr[node_type], tiles)
        n_in_tile = int(self._tile_sizes[node_type][idx])

        # the tile's own nodes lie in the margin by definition, so only the
        # neighbors' nodes need the box test
        pos = self.data[node_type]['pos'][candidates[n_in_tile:]]
        min_x, min_y, max_x, max_y = self._margin_bounds[idx]
        keep = (
            (pos[:, 0] >= min_x) & (pos[:, 0] < max_x) &
            (pos[:, 1] >= min_y) & (pos[:, 1] < max_y)
        )

        nodes = torch.cat([candidates[:n_in_tile], candidates[n_in_tile:][keep]])
        in_tile = torch.arange(nodes.numel(), device=nodes.device) < n_in_tile
        return nodes, in_tile

    def _subset_new(self, idx: int) -> Data | HeteroData:
        """Subset the Heterograph to nodes and edges within tile `idx`.

        Uses CSRs precomputed in `__init__` (`_tile_ptr`, `_neighbor_ptr`,
        `_edges_ptr`).
        """
        subset = HeteroData()
        margin_nodes = {}
        sorted_nodes = {}

        # create nodes
        for node_type in self.data.node_types:

            # get list of nodes in tile, and which of them it owns
            nodes_subset_idx, predict_mask = self._nodes_in_margin(node_type, idx)
            margin_nodes[node_type] = nodes_subset_idx

            # store sorted nodes (+ max sentinel), and order
            order = torch.argsort(nodes_subset_idx)
            max_dtype = torch.tensor(
                [torch.iinfo(nodes_subset_idx.dtype).max],
                dtype=nodes_subset_idx.dtype,
                device=nodes_subset_idx.device,
            )
            sorted_nodes[node_type] = torch.cat([nodes_subset_idx[order], max_dtype]), order

            # populate metadata for these nodes
            for key, value in self.data[node_type].items():
                if key == 'num_nodes':
                    subset[node_type].num_nodes = len(nodes_subset_idx)
                elif self.data[node_type].is_node_attr(key):
                    subset[node_type][key] = value[nodes_subset_idx]
                else:
                    subset[node_type][key] = value

            subset[node_type]['predict_mask'] = predict_mask


        # create edges
        for edge_type in self.data.edge_types:

            # get src (source) and dst (destination) nodes that fall within the tile
            src, _, dst = edge_type
            src_sorted, src_order = sorted_nodes[src]
            dst_sorted, dst_order = sorted_nodes[dst]

            # get edges where src node is in tile
            edge_src_subset_idx = query_ptr(self._edges_ptr[edge_type], margin_nodes[src])
            candidate_edges = self.data[edge_type].edge_index[:, edge_src_subset_idx]

            # find which candidate edge have a dst node in the tile
            # -> use that dst is sorted. searchsort finds the first position where it would be inserted, which
            #    is much faster than "torch.isin" for sorted arrays.
            slot = torch.searchsorted(dst_sorted, candidate_edges[1])
            edge_dst_subset_idx = dst_sorted[slot] == candidate_edges[1]

            # store mask
            kept_orig = edge_src_subset_idx[edge_dst_subset_idx]
            edge_index_new = candidate_edges[:, edge_dst_subset_idx]

            # map to new indices. "slot" is already using the subset indices, src needs to be looked up.
            edge_index_mapped = torch.stack([
                src_order[torch.searchsorted(src_sorted, edge_index_new[0])],
                dst_order[slot[edge_dst_subset_idx]],
            ], dim=0)

            # populate heterodata
            for key, value in self.data[edge_type].items():
                if key == 'edge_index':
                    subset[edge_type].edge_index = edge_index_mapped
                elif self.data[edge_type].is_edge_attr(key):
                    subset[edge_type][key] = value[kept_orig]
                else:
                    subset[edge_type][key] = value
        
        return subset

class DynamicBatchSamplerPatch(DynamicBatchSampler):
    """TODO: Description
    """
    def __len__(self):
        return len(self.dataset)  # ceiling on dataset length

