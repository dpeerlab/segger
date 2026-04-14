from torch_geometric.loader import DynamicBatchSampler
from torch_geometric.data.storage import NodeStorage
from torch_geometric.data import Data, HeteroData
from torch.utils.data import Dataset
import warnings
import shapely
import torch


from .partition import PartitionDataset
from .tiling import Tiling


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
        if isinstance(data, HeteroData):
            partition = dict()
            for node_type in data.node_types:
                partition[node_type] = self.tiling.label(
                    data[node_type][self.geometry_key]
                )
            return partition
        else:  # isinstance(data, Data)
            return self.tiling.label(data[self.geometry_key])
        
    def _mask_data(self, data: Data | HeteroData) -> Data | HeteroData:
        """
        Adds a boolean 'mask' attribute to each node indicating whether it is
        within a specified margin of a tile's boundary.
        """
        if isinstance(data, HeteroData):
            for node_type in data.node_types:
                data[node_type]['mask'] = self.tiling.mask(
                    data[node_type][self.geometry_key],
                    self.margin,
                )
        else:  # isinstance(data, Data)
            data['mask'] = self.tiling.mask(
                data[self.geometry_key],
                self.margin
            )
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
        bd_margin: float | None = None,
    ):
        """Initializes and partitions the dataset."""
        self.data = data
        self.tiling = tiling
        self.margin = float(margin)
        self.bd_margin = float(bd_margin) if bd_margin is not None else self.margin
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

        # Some datasets can carry float/non-finite edge indices after IO fallback.
        # Normalize once so PyG subgraph indexing is stable during prediction.
        if self._is_hetero:
            self._sanitize_edge_indices()

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

    def _sanitize_edge_indices(self) -> None:
        if not self._is_hetero:
            return

        for edge_type, edge_store in self.data.edge_items():
            if 'edge_index' not in edge_store:
                continue
            edge_index = edge_store.edge_index
            if edge_index is None:
                continue
            if edge_index.dim() != 2 or edge_index.size(0) != 2:
                continue
            if edge_index.numel() == 0:
                edge_store.edge_index = edge_index.to(torch.long)
                continue

            src_type, _, dst_type = edge_type
            src_n = self.data[src_type].num_nodes
            dst_n = self.data[dst_type].num_nodes
            if src_n is None:
                src_n = int(self.data[src_type]['pos'].size(0))
            if dst_n is None:
                dst_n = int(self.data[dst_type]['pos'].size(0))

            valid = torch.ones(
                edge_index.size(1),
                dtype=torch.bool,
                device=edge_index.device,
            )

            if torch.is_floating_point(edge_index):
                finite = torch.isfinite(edge_index).all(dim=0)
                rounded = edge_index.round()
                integral = torch.isclose(edge_index, rounded, rtol=0.0, atol=1e-6).all(dim=0)
                valid &= finite & integral
                edge_index = rounded

            edge_index = edge_index.to(torch.long)
            valid &= edge_index[0].ge(0) & edge_index[0].lt(int(src_n))
            valid &= edge_index[1].ge(0) & edge_index[1].lt(int(dst_n))

            if not valid.all():
                n_invalid = int((~valid).sum().item())
                n_total = int(valid.numel())
                warnings.warn(
                    "Dropping invalid edges before tile subgraph extraction "
                    f"for edge_type={edge_type} ({n_invalid}/{n_total}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                keep_idx = torch.nonzero(valid, as_tuple=False).reshape(-1)
                edge_index = edge_index[:, keep_idx]
                for attr_name in edge_store.edge_attrs():
                    if attr_name == 'edge_index':
                        continue
                    attr_value = edge_store[attr_name]
                    if (
                        isinstance(attr_value, torch.Tensor)
                        and attr_value.dim() > 0
                        and attr_value.size(0) == n_total
                    ):
                        edge_store[attr_name] = attr_value[keep_idx]

            edge_store.edge_index = edge_index

    def _subset(self, bounds: shapely.Polygon) -> Data | HeteroData:
        """Slices all node attributes within bounds.

        For heterogeneous graphs, boundary (``bd``) nodes are fetched using
        a wider margin (``bd_margin``) so that scaled polygon centroids
        remain reachable from transcript tiles.  Transcript (``tx``) nodes
        use the standard ``margin``.
        """
        inner = bounds.bounds
        outer = bounds.buffer(self.margin).bounds
        outer_bd = bounds.buffer(self.bd_margin).bounds

        if self._is_hetero:
            subset = dict()
            p_mask = dict()
            for node_type in self.data.node_types:
                pos: torch.Tensor = self.data[node_type]['pos']
                # Use wider margin for bd nodes so scaled polygon
                # centroids stay within the subgraph.
                node_outer = outer_bd if node_type == 'bd' else outer
                # Row indices of masked elements inside tile w/ margin
                subset[node_type] = (
                    (pos[:, 0] >= node_outer[0]) &
                    (pos[:, 0] <  node_outer[2]) &
                    (pos[:, 1] >= node_outer[1]) &
                    (pos[:, 1] <  node_outer[3])
                ).nonzero().flatten()
                p_mask[node_type] = (
                    (pos[subset[node_type], 0] >= inner[0]) &
                    (pos[subset[node_type], 0] <= inner[2]) &
                    (pos[subset[node_type], 1] >= inner[1]) &
                    (pos[subset[node_type], 1] <= inner[3])
                )
            sample = self.data.subgraph(subset)
            sample.set_value_dict('predict_mask', p_mask)
            sample.set_value_dict('global_index', subset)
            return sample

        else:  # is homogenous Data
            pos: torch.Tensor = self.data['pos']
            subset = (
                (pos[:, 0] >= outer[0]) &
                (pos[:, 0] <  outer[2]) &
                (pos[:, 1] >= outer[1]) &
                (pos[:, 1] <  outer[3])
            ).nonzero().flatten()
            sample = self.data.subgraph(subset)
            sample['predict_mask'] = (
                (pos[subset, 0] >= inner[0]) &
                (pos[subset, 0] <= inner[2]) &
                (pos[subset, 1] >= inner[1]) &
                (pos[subset, 1] <= inner[3])
            )
            sample['global_index'] = subset
            return sample


class DynamicBatchSamplerPatch(DynamicBatchSampler):
    """TODO: Description
    """
    def __len__(self):
        return len(self.dataset)  # ceiling on dataset length
