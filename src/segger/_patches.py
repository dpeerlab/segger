"""Workaround for pytorch/pytorch#51871 (CUDA `nonzero` INT_MAX limit).

Patches `torch_geometric.utils.bipartite_subgraph` and the references already
imported by `torch_geometric.data.hetero_data` / `._subgraph` so that
`HeteroData.subgraph` falls back to a chunked-nonzero path when the edge
tensor on CUDA exceeds INT_MAX (~2.15B) elements.

See: https://github.com/dpeerlab/segger/issues/44
"""
import torch
import torch_geometric.utils._subgraph as _sg
import torch_geometric.utils as _tgu
import torch_geometric.data.hetero_data as _hd
from torch_geometric.utils import index_to_mask
from torch_geometric.utils.map import map_index

_INT_MAX = 2**31 - 1
_pyg_bipartite = _sg.bipartite_subgraph


def chunked_nonzero(mask: torch.Tensor, chunk: int = 2**30) -> torch.Tensor:
    """Chunked version of `mask.nonzero()` that works on CUDA tensors with > INT_MAX elements."""
    if mask.numel() <= _INT_MAX or mask.device.type != "cuda":
        return mask.nonzero(as_tuple=False).flatten()
    parts = []
    for i, m in enumerate(mask.split(chunk)):
        idx = m.nonzero(as_tuple=False).flatten()
        if idx.numel():
            parts.append(idx + i * chunk)
    return torch.cat(parts)


def bipartite_safe(subset, edge_index, edge_attr=None, relabel_nodes=False,
                   size=None, return_edge_mask=False):
    """
    Replacement for `torch_geometric.utils.bipartite_subgraph`.
    Falls back to a chunked subgraph version when the edge_index is too large for CUDA.
    """
    # original
    if edge_index.numel() <= _INT_MAX or edge_index.device.type != "cuda":
        return _pyg_bipartite(subset, edge_index, edge_attr, relabel_nodes,
                              size, return_edge_mask)

    # same as source
    src_sub, dst_sub = subset
    src_mask = index_to_mask(src_sub, size=size[0])
    dst_mask = index_to_mask(dst_sub, size=size[1])
    edge_mask = src_mask[edge_index[0]] & dst_mask[edge_index[1]]

    # replaced this
    idx = chunked_nonzero(edge_mask)

    # same as source (but indices instead of mask)
    edge_index = edge_index[:, idx]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
    if relabel_nodes:
        src_index, _ = map_index(edge_index[0], src_sub, max_index=size[0], inclusive=True)
        dst_index, _ = map_index(edge_index[1], dst_sub, max_index=size[1], inclusive=True)
        edge_index = torch.stack([src_index, dst_index], dim=0)
    return (edge_index, edge_attr, edge_mask) if return_edge_mask else (edge_index, edge_attr)


_patches_applied = False


def apply():
    """Apply the patches."""
    global _patches_applied
    if _patches_applied:
        return
    _sg.bipartite_subgraph = bipartite_safe
    _tgu.bipartite_subgraph = bipartite_safe
    _hd.bipartite_subgraph = bipartite_safe
    _patches_applied = True
