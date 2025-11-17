from torch.nested._internal.nested_tensor import NestedTensor
from torch_geometric.data.storage import EdgeStorage
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, HeteroData
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal
import torch


@dataclass
class Partition:
    """Stores partition metadata for a homogeneous graph.

    This dataclass holds tensors that describe how a graph's nodes and edges
    are partitioned. It includes pointers, sizes, and the permutation used to
    sort the nodes.

    Attributes
    ----------
    node_indptr : torch.Tensor
        Index pointer for node partitions.
    edge_indptr : torch.Tensor
        Index pointer for edge partitions.
    node_sizes : torch.Tensor
        The size of each node partition.
    edge_sizes : torch.Tensor
        The size of each edge partition.
    node_permutation : torch.Tensor
        The permutation that was applied to the original nodes to sort them
        by partition.
    """
    node_indptr:        torch.Tensor = None
    edge_indptr:        torch.Tensor = None
    node_sizes:         torch.Tensor = None
    edge_sizes:         torch.Tensor = None
    node_permutation:   torch.Tensor = None

    def _validate_num_partitions(self) -> bool:
        """Confirms all node and edge elements have same numbers of partitions.
        """
        node_attributes = [self.node_sizes, self.node_indptr]
        if not any(node_attributes):
            return True
        elif not all(node_attributes):
            return False
        
        edge_attributes = [self.edge_sizes, self.edge_indptr]
        if not any(edge_attributes):
            return True
        elif not all(edge_attributes):
            return False

        # Check for consistent lengths.
        num_partitions = [
            len(self.node_sizes),
            len(self.edge_sizes),
            len(self.node_indptr) - 1,
            len(self.edge_indptr) - 1,
        ]
        
        return len(set(num_partitions)) == 1
        
    def __len__(self) -> int:
        """
        Returns number of partitions tracked by this partitioning, 0 if empty.
        """
        if not self._validate_num_partitions():
            raise ValueError(
                "This `Partition` contains inconsistent numbers of partitions "
                "across elements."
            )
        if self.node_sizes:
            return len(self.node_sizes)
        if self.edge_sizes:
            return len(self.edge_sizes)
        return 0


@dataclass
class HeteroPartition:
    """Stores partition metadata for a heterogeneous graph.

    This dataclass holds dictionaries that map node/edge types to their
    respective partition metadata tensors.

    Attributes
    ----------
    node_indptr : dict
        Maps node type to its node index pointer tensor.
    edge_indptr : dict
        Maps edge type to its edge index pointer tensor.
    node_sizes : dict
        Maps node type to its node partition sizes tensor.
    edge_sizes : dict
        Maps edge type to its edge partition sizes tensor.
    node_permutation : dict
        Maps node type to the permutation tensor that was applied to its nodes.
    """
    node_indptr:        dict = field(default_factory=dict)
    edge_indptr:        dict = field(default_factory=dict)
    node_sizes:         dict = field(default_factory=dict)
    edge_sizes:         dict = field(default_factory=dict)
    node_permutation:   dict = field(default_factory=dict)

    def _validate_keys(self) -> bool:
        """Confirms all node and edge elements have same sets of keys."""
        node_sets = [
            set(self.node_sizes),
            set(self.node_indptr),
            set(self.node_permutation),
        ]
        edge_sets = [
            set(self.edge_sizes),
            set(self.edge_indptr),
        ]
        return all(s == node_sets[0] for s in node_sets) and \
               all(s == edge_sets[0] for s in edge_sets)
    
    def _validate_num_partitions(self) -> bool:
        """Confirms all node and edge elements have same numbers of partitions.
        """
        node_attributes = [self.node_sizes.values(), self.node_indptr.values()]
        if not any(node_attributes):
            return True
        elif not all(node_attributes):
            return False

        edge_attributes = [self.edge_sizes.values(), self.edge_indptr.values()]
        if not any(edge_attributes):
            return True
        elif not all(edge_attributes):
            return False
        num_partitions = [
            *[len(v) for v in self.node_sizes.values()],
            *[len(v) for v in self.edge_sizes.values()],
            *[len(v) - 1 for v in self.node_indptr.values()],
            *[len(v) - 1 for v in self.edge_indptr.values()],
        ]

        return len(set(num_partitions)) == 1
    
    def __len__(self) -> int:
        """
        Returns number of partitions tracked by this partitioning, 0 if empty.
        """
        if not self._validate_num_partitions():
            raise ValueError(
                "This `HeteroPartition` contains inconsistent numbers of "
                "partitions across elements."
            )
        if self.node_sizes:
            return len(next(iter(self.node_sizes.values())))
        if self.edge_sizes:
            return len(next(iter(self.edge_sizes.values())))
        return 0



class PartitionDataset(torch.utils.data.Dataset):
    """Represents a PyG dataset partitioned into disconnected subgraphs.

    This class takes a PyG `Data` or `HeteroData` object and a partition
    definition, producing a new graph where nodes and edges are permuted
    according to the specified partitions.

    Parameters
    ----------
    data : Data or HeteroData
        The input graph to partition.
    partition : Any
        The partition definition.
        - For dense layout, this is a `torch.Tensor` (for homogeneous
          graphs) or a `dict[str, torch.Tensor]` (for heterogeneous
          graphs) containing partition labels for each node.
        - For sparse layout, this is a `Partition` or `HeteroPartition`
          object that already describes the partitioned graph.

    Attributes
    ----------
    data : Data or HeteroData
        The new, permuted graph object.
    partition : Partition or HeteroPartition
        An object containing all metadata about the partitions.
    """
    def __init__(
        self,
        data: Data | HeteroData,
        partition: Any,
        clone: bool = True,
        transform: BaseTransform = None,
    ):
        """Initializes and partitions the dataset."""
        self._is_hetero = isinstance(data, HeteroData)
        self._is_sparse = isinstance(partition, (Partition, HeteroPartition))

        if self._is_sparse:
            self._validate_sparse(data, partition)
            self.data = data.clone() if clone else data
            self.partition = partition
            self._num_partitions = len(self.partition)

        else:
            self._validate_dense(data, partition)
            self.partition = (
                HeteroPartition() if self._is_hetero else Partition())
            self.data = data.clone() if clone else data
            # Calculate global no. partitions upfront
            if self._is_hetero:
                self._num_partitions = -1
                for labels in partition.values():
                    if labels.numel() > 0:
                        self._num_partitions = max(
                            self._num_partitions,
                            labels.max().item()
                        )
                self._num_partitions += 1
            else:
                self._num_partitions = partition.max().item() + 1
            self._permute_nodes(partition)
            self._permute_edges(partition)

        # Allow for graph transforms when retrieving items
        self.transform = transform

    def _validate_sparse(self, data: Data | HeteroData, partition: Any):
        """
        Validates that a sparse partition object is consistent with the graph.
        """
        if self._is_hetero:
            if not isinstance(partition, HeteroPartition):
                raise ValueError(
                    "For a heterogeneous graph, sparse input must be a "
                    "`HeteroPartition` object."
                )
            if not partition._validate_keys():
                raise ValueError(
                    "Provided `HeteroPartition` contains inconsistent node "
                    "or edge keys across elements."
                )
            if not partition._validate_num_partitions():
                raise ValueError(
                    "Provided `HeteroPartition` contains inconsistent numbers "
                    "of partitions across elements."
                )
            for attr, store in partition.__dict__.items():
                if not isinstance(store, dict):
                    raise TypeError(
                        f"Attribute '{attr}' in `HeteroPartition` must be a "
                        f"dict."
                    )
                is_node_attr = all(n in store for n in data.node_types)
                is_edge_attr = all(e in store for e in data.edge_types)
                if not (is_node_attr or is_edge_attr):
                    raise KeyError(
                        f"Dictionary for `HeteroPartition` attribute '{attr}' "
                        f"is missing keys for the graph's node or edge types."
                    )
                for key, value in store.items():
                    if not isinstance(value, torch.Tensor):
                        raise TypeError(
                            f"Value for '{key}' in attribute '{attr}' must be "
                            f"a `torch.Tensor`, not `{type(value).__name__}`."
                        )
        else:  # homogeneous graph
            if not isinstance(partition, Partition):
                raise ValueError(
                    "For a homogeneous graph, sparse input must be a "
                    "`Partition` object."
                )
            if not partition._validate_num_partitions():
                raise ValueError(
                    "Provided `Partition` contains inconsistent numbers of "
                    "partitions across elements."
                )
            for attr, store in partition.__dict__.items():
                if not isinstance(store, torch.Tensor):
                    raise TypeError(
                        f"Value for attribute '{attr}' in partition must be a "
                        f"`torch.Tensor`, not `{type(store).__name__}`."
                    )

    def _validate_dense(self, data: Data | HeteroData, partition: Any):
        """
        Validates that a dense partition input is consistent with the graph.
        """
        if self._is_hetero:
            if not isinstance(partition, dict):
                raise TypeError(
                    "For a heterogeneous graph, dense input must be a "
                    "dictionary mapping node types to partition labels."
                )
            for node_type in data.node_types:
                if node_type not in partition:
                    raise KeyError(
                        f"The `partition` dictionary is missing an entry for "
                        f"node type: '{node_type}'."
                    )
                if not isinstance(partition[node_type], torch.Tensor):
                    raise TypeError(
                        f"The partition for node type '{node_type}' must be a "
                        f"`torch.Tensor`, not {type(partition[node_type])}."
                    )
        elif not isinstance(partition, torch.Tensor):
            raise TypeError(
                "For a homogeneous graph, dense input must be a `torch.Tensor`."
            )
        
    @staticmethod
    def _index_select(
        input: torch.Tensor,
        index: torch.Tensor | int | slice,
        dim: int = 0,
    ):
        # Dense tensor; all tensors support integer indexing
        if input.layout == torch.strided or isinstance(index, int):
            return input[index]
        if isinstance(index, slice):
            start = 0 if index.start is None else index.start
            step = 1 if index.step is None else index.step
            index = torch.arange(start, index.stop, step)
        # Sparse Tensor, only supports index_select
        if input.layout == torch.sparse_coo:
            return torch.index_select(input, dim, index)
        # Nested Tensor, need to rebuild from list comprehension
        elif input.layout == torch.jagged:
            if len(index) == 0:
                return
            input_list = input.unbind()
            return torch.nested.nested_tensor(
                [input_list[i] for i in index],
                layout=input.layout,
            )
        else:
            raise ValueError(
                f"Unsupported layout '{input.layout}'. Tensor layout must be "
                f"one of 'strided', 'sparse_coo', or 'jagged'."
            )

    def _permute_nodes(
        self,
        labels: torch.Tensor | dict[str, torch.Tensor],
    ):
        """Permutes all node attributes and stores partition metadata.
        
        This method iterates through each node type, calculates the correct
        node permutation based on the partition labels, applies this
        permutation to all node-level attributes, and saves the resulting
        metadata to `self.partition`.
        """
        if self._is_hetero:
            # Get partition info for node type and permute its attributes
            for node in self.data.node_types:
                (
                    self.partition.node_permutation[node],
                    self.partition.node_indptr[node],
                    self.partition.node_sizes[node],
                ) = self._permute_node_labels(labels[node])
                for attr in self.data[node].node_attrs():
                    self.data[node][attr] = self._index_select(
                        self.data[node][attr],
                        self.partition.node_permutation[node],
                        dim=0,
                    )
        else:
            (
                self.partition.node_permutation,
                self.partition.node_indptr,
                self.partition.node_sizes,
            ) = self._permute_node_labels(labels)
            for attr in self.data.node_attrs():
                self.data[attr] = self._index_select(
                    self.data[attr],
                    self.partition.node_permutation,
                    dim=0,
                )
        
    def _permute_node_labels(
        self,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates the permutation and pointers for a set of node labels."""
        # Get permutation to sort nodes by partition
        permutation = torch.argsort(labels)

        # Pointers to splits between partitions
        sizes = torch.bincount(
            labels[permutation],
            minlength=self._num_partitions,
        )
        indptr = torch.cat((
            torch.tensor([0], device=labels.device),
            torch.cumsum(sizes, dim=0)
        ))
        return permutation, indptr, sizes
    
    def _permute_edges(
        self,
        labels: torch.Tensor | dict[str, torch.Tensor],
    ):
        """Permutes all edge attributes and stores partition metadata.

        This method iterates through each edge type, calculates the correct
        edge permutation based on the source nodes' partitions, applies this
        permutation to all edge-level attributes, and saves the resulting
        metadata to `self.partition`.
        """
        if self._is_hetero:
            for edge_type in self.data.edge_types:
                src, _, dst = edge_type
                (
                    self.partition.edge_indptr[edge_type],
                    self.partition.edge_sizes[edge_type],
                ) = self._permute_edge_store(
                    self.data[edge_type],
                    self.partition.node_permutation[src],
                    self.partition.node_permutation[dst],
                    labels[src],
                    labels[dst],
                )
        else:
            (
                self.partition.edge_indptr,
                self.partition.edge_sizes,
            ) = self._permute_edge_store(
                self.data,
                self.partition.node_permutation,
                self.partition.node_permutation,
                labels,
                labels,
            )

    def _map_edge_index(
        self,
        edge_store: Data | EdgeStorage,
        src_perm: torch.Tensor,
        dst_perm: torch.Tensor,
    ):
        """Remaps the `edge_index` to the new permuted node indices."""
        # Map edge index and attributes to new indices
        inv_src_perm = torch.empty_like(src_perm)
        inv_src_perm[src_perm] = torch.arange(
            src_perm.numel(),
            device=src_perm.device,
        )
        inv_dst_perm = torch.empty_like(dst_perm)
        inv_dst_perm[dst_perm] = torch.arange(
            dst_perm.numel(),
            device=dst_perm.device,
        )
        edge_store.edge_index = torch.stack([
            inv_src_perm[edge_store.edge_index[0]],
            inv_dst_perm[edge_store.edge_index[1]],
        ])

    def _permute_edge_store(
        self,
        edge_store: Data | EdgeStorage,
        src_perm: torch.Tensor,
        dst_perm: torch.Tensor,
        src_labels: torch.Tensor,
        dst_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates and applies the permutation for a single edge type."""
        # Map edge index to new indices
        self._map_edge_index(edge_store, src_perm, dst_perm)

        # Get permutation to sort edges by src partition
        src_edge_labels = src_labels[src_perm][edge_store.edge_index[0]]
        dst_edge_labels = dst_labels[dst_perm][edge_store.edge_index[1]]
        permutation = torch.argsort(src_edge_labels)

        # Get mask over inter-partition edges
        src_edge_labels = src_edge_labels[permutation]
        dst_edge_labels = dst_edge_labels[permutation]
        mask = src_edge_labels == dst_edge_labels

        # Update edge store with permutation, including edge index
        for attr in edge_store.edge_attrs():
            if attr == 'edge_index':
                edge_store[attr] = edge_store[attr][:, permutation][:, mask]
            else:
                edge_store[attr] = self._index_select(
                    edge_store[attr],
                    permutation[mask],
                    dim=0,
                )
        
        # Get partition properties
        sizes = torch.bincount(
            src_edge_labels[mask],
            minlength=self._num_partitions,
        )
        indptr = torch.cat((
            torch.tensor([0], device=src_edge_labels.device),
            torch.cumsum(sizes, dim=0),
        ))

        return indptr, sizes

    def __len__(self) -> int:
        """Description."""
        return self._num_partitions
    
    def __getitem__(self, index: int):
        """Get the graph partition associated at location `index`. 
        
        Initializes an empty Data or HeteroData object and populates with node
        and edge attributes associated with the indexed graph partition. Other
        non-node/edge attributes are populated without subsetting.
        """
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(
                f"Index {index} is out of range for dataset with {len(self)} "
                f"partitions."
            )
        if self._is_hetero:
            part = HeteroData()
            for node_type, node_store in self.data.node_items():
                node_i = self.partition.node_indptr[node_type][index]
                node_j = self.partition.node_indptr[node_type][index + 1]
                for name, attr in node_store.items():
                    if node_store.is_node_attr(name):
                        part[node_type][name] = self._index_select(
                            attr,
                            slice(node_i, node_j),
                        )
                    else:
                        part[node_type][name] = attr
            for edge_type, edge_store in self.data.edge_items():
                src, _, dst = edge_type
                node_i_src = self.partition.node_indptr[src][index]
                node_i_dst = self.partition.node_indptr[dst][index]
                edge_i = self.partition.edge_indptr[edge_type][index]
                edge_j = self.partition.edge_indptr[edge_type][index + 1]
                for name, attr in edge_store.items():
                    if name == 'edge_index':
                        edge_index = attr[:, edge_i:edge_j].clone()
                        edge_index[0] -= node_i_src
                        edge_index[1] -= node_i_dst
                        part[edge_type][name] = edge_index
                    elif edge_store.is_edge_attr(name):
                        part[edge_type][name] = self._index_select(
                            attr,
                            slice(edge_i, edge_j),
                        )
                    else:
                        part[edge_type][name] = attr
        else:
            part = Data()
            for name, attr in self.data:
                node_i, node_j = self.partition.node_indptr[index:index + 2]
                edge_i, edge_j = self.partition.edge_indptr[index:index + 2]
                if self.data.is_node_attr(name):
                    part[name] = self._index_select(attr, slice(node_i, node_j))
                elif self.data.is_edge_attr(name):
                    if name == 'edge_index':
                        edge_index = attr[:, edge_i:edge_j].clone()
                        part[name] = edge_index - node_i
                    else:
                        part[name] = self._index_select(
                            attr,
                            slice(edge_i, edge_j)
                        )
                else:
                    part[name] = attr
        # Optionally transform
        part = part if self.transform is None else self.transform(part)
        
        return part

    def _add_node_attr(
        self,
        key: str,
        attr: torch.Tensor,
        node_type: str = None,
    ):
        """Adds and permutes a new node attribute to self.data.

        The provided attribute tensor must correspond to the nodes in the 
        original graph before partitioning.

        Parameters
        ----------
        key : str
            The name for the new node attribute.
        attr : torch.Tensor
            An attribute tensor whose ordering corresponds to the nodes in the
            original, un-partitioned graph.
        node_type : str, optional
            The target node type for the attribute. This is required for
            heterogeneous graphs.
        
        Raises
        ------
        ValueError
            If `node_type` is omitted for a heterogeneous graph, if the `attr` 
            tensor is too small for the permutation, or if `node_type` is 
            provided for a homogeneous graph.
        """
        if self._is_hetero:
            if node_type is None:
                raise ValueError(
                    "A node type must be supplied for HeteroData attributes."
                )
            node_perm  = self.partition.node_permutation[node_type]
            node_store = self.data[node_type]
        else:
            if node_type is not None:
                raise ValueError(
                    "No node type should be supplied for Data attributes."
                )
            node_perm = self.partition.node_permutation
            node_store = self.data
        if attr.shape[0] > node_perm.max() + 1:
            raise ValueError(
                f"Attribute tensor of shape {attr.shape} is larger than the "
                f"number of nodes in the permutation: {node_perm.max() + 1}."
            )
        node_store[key] = self._index_select(attr, node_perm)
