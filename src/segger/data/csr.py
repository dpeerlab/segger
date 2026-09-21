"""CSR helpers for grouping elements by bucket at billion-element scale.

Conventions
-----------
* `ptr` is int64 with `n_buckets + 1` entries: bucket `i` has `values[ptr[i]:ptr[i + 1]]`.
* `values` is int32 while the input is shorter than 2**31, int64 above.
* Values ascend within each bucket, matching a stable sort by label.
"""

from torch_geometric.index import index2ptr
import torch


INT_MAX = 2 ** 31 - 1
CSR_CHUNK = 2 ** 24 # <- up to 0.8GB per chunk, keeping it small
LOOP_MAX_BUCKETS = 256 # <- past this the slicing loop costs more than it saves


def query_ptr(
    csr: tuple[torch.Tensor, torch.Tensor],
    query: int | torch.Tensor,
    total: int | None = None,
) -> torch.Tensor:
    """Gather values for bucket(s) `query` from a `(ptr, values)` CSR.

    `query` may be a scalar (one bucket) or a 1-D tensor (concatenated in
    the given order). `total` is how many values the query covers, counted
    here when not given.

    A short query slices its buckets and concatenates them, a long one
    builds the index arrays instead; profiling put the switch past 256
    buckets, where the Python loop starts costing more than the ~200 us
    floor of the vectorised path.
    """
    ptr, values = csr

    # single value
    if not (torch.is_tensor(query) and query.dim() > 0):
        q = int(query)
        return values[ptr[q]:ptr[q + 1]]

    if query.numel() == 0:
        return values.new_empty(0)

    # tensor of values
    starts = ptr[query]
    ends = ptr[query + 1]

    # few buckets: slice each bucket individually
    if query.numel() < LOOP_MAX_BUCKETS:
        spans = torch.stack((starts, ends)).tolist()
        return torch.cat([values[s:e] for s, e in zip(*spans)])

    # vectorised for querying many
    counts = ends - starts
    if total is None:
        total = int(counts.sum())
    if total == 0:
        return values.new_empty(0)
    base = torch.repeat_interleave(starts, counts, output_size=total)
    within = (torch.arange(total, device=values.device) - torch.repeat_interleave(
        counts.cumsum(0) - counts, counts, output_size=total))
    return values[base + within]


def _count_ptr(
    labels: torch.Tensor,
    n_buckets: int,
    chunk: int,
) -> torch.Tensor:
    """Bucket boundaries from a chunked histogram, helps with sorting indices larger than > 2^32-1."""
    counts = torch.zeros(n_buckets, dtype=torch.int64, device=labels.device)
    for start in range(0, labels.numel(), chunk):
        counts += torch.bincount(labels[start:start + chunk], minlength=n_buckets)
    return torch.cat([
        torch.zeros(1, dtype=torch.int64, device=labels.device),
        counts.cumsum(0),
    ])


def _argsort_chunked(
    labels: torch.Tensor,
    n_buckets: int,
    cursor: torch.Tensor,
    dtype: torch.dtype,
    chunk: int,
) -> torch.Tensor:
    """Stable argsort for inputs over INT_MAX elements. Uses ptr for boundaries.
    """
    n = labels.numel()
    device = labels.device

    # place each chunk's elements, bucket by bucket
    cursor = cursor.clone()
    values = torch.empty(n, dtype=dtype, device=device)
    for start in range(0, n, chunk):

        # get chunk of labels
        labels_chunk = labels[start:start + chunk]
        counts_chunk = torch.bincount(labels_chunk, minlength=n_buckets)

        # get indices, sorted by bucket: position within the chunk's own run
        order = torch.argsort(labels_chunk, stable=True)
        run_start = torch.repeat_interleave(counts_chunk.cumsum(0) - counts_chunk, counts_chunk)
        within = torch.arange(labels_chunk.numel(), device=device) - run_start

        # update values (chunked), offset by where each bucket stands globally
        values[torch.repeat_interleave(cursor, counts_chunk) + within] = ((order + start).to(dtype))

        # update cursor
        cursor += counts_chunk

    return values


def index_to_ptr(
    labels: torch.Tensor,
    is_sorted: bool = False,
    n_buckets: int | None = None,
    chunk: int = CSR_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable csr-style data preparation, supporting arrays larger than 2**31 - 1 elements.

    Adjustments:
    * Using chunked sorting, since argsort supports atmost 2*31-1 values
      and peaks at ~6x the input in workspace.
    * `index2ptr` only runs on an already sorted index, so unsorted labels
      are counted in chunks instead.

    Parameters
    ----------
    labels : 1-D bucket label per element.
    is_sorted : Whether `labels` already ascends, leaving nothing to permute.
    n_buckets : Bucket count; `labels.max() + 1` when None, which drops empty buckets.
    chunk : Elements per sorting pass.

    Example:
    ```
    labels = torch.tensor([0, 1, 0, 2, 1])
    ptr, values = index_to_ptr(labels, n_buckets=3)
    ```

    Output:
    - ptr: tensor([0, 2, 4, 5])
    - values: tensor([0, 2, 1, 4, 3])

    where `ptr[0]:ptr[1]` gives the indices of elements in bucket 0, `ptr[1]:ptr[2]` for bucket 1, and so on
    and `values[ptr[i]:ptr[i + 1]]` gives the original indices of elements in bucket `i`.
    """

    # assertions
    if labels.dim() != 1:
        raise ValueError(f"'labels' must be 1-D, but got {labels.shape}.")
    n = labels.numel()
    if n and int(labels.min()) < 0:
        raise ValueError(
            f"{int((labels < 0).sum())}/{n} labels are negative (unassigned); "
            f"every node must fall in a bucket."
        )

    # rough test if the first 10k elements ascend
    if is_sorted and n > 1 and int(labels[:10000].diff().min()) < 0:
        raise ValueError("Elements of 'labels' are not sorted.")

    # initialise
    dtype = torch.int32 if n < 2 ** 31 else torch.int64
    if n_buckets is None:
        n_buckets = int(labels.max()) + 1 if n else 0

    # skip sorting if already sorted
    if is_sorted:
        if labels.dtype != torch.int64 and n > INT_MAX:
            labels = labels.long()
        ptr = index2ptr(labels, n_buckets).to(torch.int64)
        return ptr, torch.arange(n, dtype=dtype, device=labels.device)

    # always use chunked argsort which needs less memory
    ptr = _count_ptr(labels, n_buckets, chunk)
    return ptr, _argsort_chunked(labels, n_buckets, ptr[:-1], dtype, chunk)
