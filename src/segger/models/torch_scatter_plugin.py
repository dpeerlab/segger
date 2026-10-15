"""Stand-in for `torch_scatter.scatter_max` built on native torch ops.

torch-scatter does not (yet) publish a wheel for torch 2.14. Thus, need to replace
this module until a later version is available:
https://github.com/rusty1s/pytorch_scatter/issues/513

Matches torch_scatter on ordinary input, empty buckets, ties, and a single edge.
"""
import torch


def scatter_max(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce `src` to a per-bucket maximum and the position that produced it.

    Only supports a 1-D case segger, not the full `torch_scatter.scatter_max`. 
    Ties resolve to the lowest position, where torch_scatter's atomics leave the winner undefined.

    Parameters
    ----------
    src : torch.Tensor
        Values to reduce, one per edge.
    index : torch.Tensor
        Bucket each value belongs to, in `[0, dim_size)`.
    dim_size : int
        Number of buckets in the output.

    Returns
    -------
    tuple of torch.Tensor
        Per-bucket maximum, and the index into `src` that produced it.
    """
    # initialize output tensors
    n_src = src.numel()
    values = src.new_full((dim_size,), float('-inf'))
    values.scatter_reduce_(0, index, src, reduce='amax', include_self=False)

    # get positions of the maximum values
    positions = torch.arange(n_src, device=src.device)
    hit = src == values[index]
    argmax = positions.new_full((dim_size,), n_src)
    argmax.scatter_reduce_(
        0, index[hit], positions[hit], reduce='amin', include_self=False
    )
    return torch.where(values.isinf(), 0.0, values), argmax
