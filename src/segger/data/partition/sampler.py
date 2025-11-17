from typing import List, Literal, Iterator, Optional
from torch_geometric.loader import DataLoader
import random
import torch
import math


from .dataset import PartitionDataset


def best_fit_decreasing(
    items: List[float],
    bin_capacity: float,
    skip_too_big: bool = False,
) -> List[List[int]]:
    """Implements the Best-Fit Decreasing (BFD) bin packing algorithm.

    BFD works by first sorting all items from largest to smallest. Then, each 
    item is placed into the bin where it fits most tightly (i.e., the bin with 
    the least remaining capacity that can still hold the item).

    Parameters
    ----------
    items : list of float
        A list of the sizes of items to be packed.
    bin_capacity : float
        The capacity of each bin.
    skip_too_big : bool, optional
        If True, items larger than `bin_capacity` or <= 0 are ignored instead 
        of raising an error. Defaults to False.

    Returns
    -------
    list of list of int
        A list of bins, where each bin is a list of the original indices of the 
        items it contains.

    Raises
    ------
    ValueError
        If any item has a size greater than `bin_capacity` or less than or 
        equal to 0, and `skip_too_big` is False.

    """
    if skip_too_big:
        indexed_items = [
            (val, i) for i, val in enumerate(items)
            if 0 < val <= bin_capacity
        ]
    else:
        if not all(0 < item <= bin_capacity for item in items):
            raise ValueError(
                "All items must be > 0 and <= bin_capacity."
            )
        indexed_items = [(val, i) for i, val in enumerate(items)]

    # Sort items by size in descending order.
    indexed_items.sort(key=lambda x: x[0], reverse=True)

    bins: List[List[int]] = []
    bin_capacities: List[float] = []

    for item_val, item_idx in indexed_items:
        best_bin_idx = -1
        min_remaining_space = float('inf')
        # Find the best bin for the current item.
        for i, capacity in enumerate(bin_capacities):
            if capacity >= item_val:
                remaining_space = capacity - item_val
                if remaining_space < min_remaining_space:
                    min_remaining_space = remaining_space
                    best_bin_idx = i
        # If a suitable bin was found, place the item there.
        if best_bin_idx != -1:
            bins[best_bin_idx].append(item_idx)
            bin_capacities[best_bin_idx] -= item_val
        # Otherwise, open a new bin for the item.
        else:
            bins.append([item_idx])
            bin_capacities.append(bin_capacity - item_val)

    return bins


def harmonic_k(
    items: List[float],
    bin_capacity: float,
    k: int = 6,
    skip_too_big: bool = False,
) -> List[List[int]]:
    """Implements the Harmonic-k online bin packing algorithm.

    Classifies each incoming item into a harmonic interval based on its size 
    and packs it with other items from the same interval. It processes items 
    in the order they arrive.

    The `k` parameter defines `k-1` intervals for items > 1/k, while 
    items <= 1/k are treated as "small" and packed together.

    Parameters
    ----------
    items : list of float
        A list of the sizes of items to be packed.
    bin_capacity : float
        The capacity of each bin.
    k : int, optional
        The integer defining the harmonic intervals. Must be >= 2.
        Defaults to 6.
    skip_too_big : bool, optional
        If True, items larger than `bin_capacity` or <= 0 are ignored instead 
        of raising an error. Defaults to False.

    Returns
    -------
    list of list of int
        A list of bins, where each bin is a list of the original indices of the 
        items it contains.

    Raises
    ------
    ValueError
        If an invalid item size is found and `skip_too_big` is False, or if `k` 
        is less than 2.

    """
    if k < 2:
        raise ValueError("Parameter k must be an integer >= 2.")

    if skip_too_big:
        indexed_items = [
            (val, i) for i, val in enumerate(items)
            if 0 < val <= bin_capacity
        ]
    else:
        if not all(0 < item <= bin_capacity for item in items):
            raise ValueError(
                "All items must be > 0 and <= bin_capacity."
            )
        indexed_items = list(enumerate(items))
        indexed_items = [(val, i) for i, val in indexed_items]

    # Finalized bins
    bins: List[List[int]] = []
    open_bins: dict[int, List[int]] = {}

    # Small items (<= 1/k) are packed separately
    small_item_bins: List[List[int]] = []
    small_item_capacities: List[float] = []

    for item_val, item_idx in indexed_items:
        scaled_val = item_val / bin_capacity
        if scaled_val > 1 / k:
            # Item belongs to a harmonic interval j
            j = math.floor(1 / scaled_val)
            if j not in open_bins:
                open_bins[j] = []
            open_bins[j].append(item_idx)
            # If a bin is full (contains j items of type j), finalize it
            if len(open_bins[j]) == j:
                bins.append(open_bins[j])
                open_bins[j] = []
        else:
            # Pack small items using First-Fit
            placed = False
            for i, capacity in enumerate(small_item_capacities):
                if item_val <= capacity:
                    small_item_bins[i].append(item_idx)
                    small_item_capacities[i] -= item_val
                    placed = True
                    break
            if not placed:
                small_item_bins.append([item_idx])
                small_item_capacities.append(bin_capacity - item_val)

    # Add any partially-filled harmonic bins to the final list
    for j in open_bins:
        if open_bins[j]:
            bins.append(open_bins[j])

    # Add the bins containing small items
    bins.extend(small_item_bins)

    return bins


def first_fit_decreasing_bucketed(
    items: List[float],
    bin_capacity: float,
    skip_too_big: bool = False,
    n_buckets: Optional[int] = 1,
    rng: Optional[random.Random] = None,
) -> List[List[int]]:
    """Implements FFD with optional value-clustered randomization.

    After the usual descending sort, values can be divided into `n_buckets`
    contiguous buckets. Items are shuffled only inside each bucket; the
    bucket order is preserved. The packing uses the First-Fit heuristic.

    Parameters
    ----------
    items : list of float
        A list of the sizes of items to be packed.
    bin_capacity : float
        The capacity of each bin.
    skip_too_big : bool, optional
        If True, items larger than `bin_capacity` or <= 0 are ignored instead
        of raising an error. Defaults to False.
    n_buckets : int, optional
        - If None or >= len(items), deterministic FFD is performed.
        - If 1, a fully random First-Fit is performed.
        - If >= 2, shuffles items inside `n_buckets` value-based buckets.
    rng : random.Random, optional
        Instance of `random.Random` for reproducibility. If None, the global
        RNG is used.

    Returns
    -------
    list of list of int
        A list of bins, where each bin is a list of the original indices of the
        items it contains.

    Raises
    ------
    ValueError
        If any item has a size greater than `bin_capacity` or less than or
        equal to 0, and `skip_too_big` is False.

    """
    rng = rng or random

    if skip_too_big:
        indexed_items = [
            (val, i) for i, val in enumerate(items)
            if 0 < val <= bin_capacity
        ]
    else:
        if not all(0 < item <= bin_capacity for item in items):
            raise ValueError(
                "All items must be > 0 and <= bin_capacity."
            )
        indexed_items = [(val, i) for i, val in enumerate(items)]

    if not indexed_items:
        return []

    # Sort items by size in descending order.
    indexed_items.sort(key=lambda x: x[0], reverse=True)

    # Optional: Shuffle items within value-based buckets.
    n = len(indexed_items)
    if n_buckets is not None and 1 <= n_buckets < n:
        if n_buckets == 1:
            rng.shuffle(indexed_items)  # Full shuffle
        else:
            # Find positions of the (k-1) largest adjacent gaps.
            gaps = [
                (indexed_items[i - 1][0] - indexed_items[i][0], i)
                for i in range(1, n)
            ]
            cut_at = {
                pos for _, pos in sorted(gaps, reverse=True)[:n_buckets - 1]
            }

            # Shuffle within each bucket.
            start = 0
            for i in range(1, n + 1):
                if i in cut_at or i == n:
                    rng.shuffle(indexed_items[start:i])
                    start = i

    bins: List[List[int]] = []
    bin_capacities: List[float] = []

    for item_val, item_idx in indexed_items:
        placed_in_bin = False
        # Find the first bin that can hold the item.
        for i, capacity in enumerate(bin_capacities):
            if capacity >= item_val:
                bins[i].append(item_idx)
                bin_capacities[i] -= item_val
                placed_in_bin = True
                break
        
        # If no suitable bin was found, open a new one.
        if not placed_in_bin:
            bins.append([item_idx])
            bin_capacities.append(bin_capacity - item_val)

    return bins


class PartitionSampler(torch.utils.data.Sampler):
    """A batch sampler that packs data partitions into pre-computed batches.

    This sampler groups partitions (e.g., subgraphs) into batches using bin 
    packing algorithms. Batches are pre-computed to ensure the sampler's length 
    is always accurate.

    If `shuffle` is True, it uses an online algorithm (Harmonic-k) and
    regenerates the batches with a new shuffle after iterating through once
    (e.g., at the beginning of a new epoch). If `shuffle` is False, it uses an 
    offline algorithm (BFD) and computes the batches only once.
    """
    def __init__(
        self,
        dataset: PartitionDataset,
        max_num: int,
        mode: Literal["node", "edge"] = "edge",
        subset: list[int] = None,
        shuffle: bool = False,
        skip_too_big: bool = False,
    ):
        """Initializes the DynamicPartitionBatchSampler.

        Parameters
        ----------
        dataset : PartitionDataset
            The dataset containing the partitions to be batched.
        max_num : int
            The maximum number of nodes or edges allowed per batch.
        mode : {"node", "edge"}, optional
            Determines whether to use partition node counts or edge counts as 
            the weights for packing. Defaults to "edge".
        subset : list of int, optional
            A list of partition indices to sample from. If None, the entire
            dataset is used. Defaults to None.
        shuffle : bool, optional
            If True, the partitions are shuffled at the start of each epoch,
            and an online packing algorithm is used to create different
            batches. If False, an offline algorithm is used to create a
            fixed, deterministic set of batches. Defaults to False.
        skip_too_big : bool, optional
            If True, partitions larger than `max_num` are ignored. If False, 
            the packing algorithm will raise a ValueError. Defaults to False.
        """
        self.dataset = dataset
        self.max_num = max_num
        self.mode = mode
        self.subset = subset
        self.shuffle = shuffle
        self.skip_too_big = skip_too_big
        self.packing_algo = (
            first_fit_decreasing_bucketed if self.shuffle else
            best_fit_decreasing
        )

        # Get partition sizes in numbers of nodes or edges
        if mode == "edge":
            weights = self.dataset.partition.edge_sizes
        else:
            weights = self.dataset.partition.node_sizes
        
        if self.dataset._is_hetero:
            weights = torch.stack(list(weights.values())).sum(dim=0)
        self.weights = weights.tolist()

        # Start as stale in case something goes wrong in batch generation
        self.batches: List[List[int]] = []
        self.stale: bool = True

        # Pre-compute batches
        self._generate_batches()

    def _generate_batches(self) -> None:
        """Generates and stores a new set of batches for an epoch."""
        indices = self.subset if self.subset is not None else \
                  list(range(len(self.weights)))

        if self.shuffle:
            random.shuffle(indices)

        # Get new packed bins
        weights_to_pack = [self.weights[i] for i in indices]
        bins = self.packing_algo(
            weights_to_pack,
            self.max_num,
            skip_too_big=self.skip_too_big,
        )

        # Map local indices from the packer back to original dataset indices
        self.batches = [[indices[i] for i in bin] for bin in bins]
        self.stale = False

    def __iter__(self) -> Iterator[list[int]]:
        """Yields batches of size <= `self.max_num`.

        If shuffling is enabled and the current batches are stale (e.g., from a
        previous epoch), this method triggers a regeneration of batches.
        """
        if self.stale:
            self._generate_batches()
        for batch in self.batches:
            yield batch
        if self.shuffle:
            self.stale = True

    def __len__(self) -> int:
        """Returns the total number of batches for the upcoming epoch.

        If batches have not been computed yet, this method will trigger
        their generation to ensure the length is accurate.
        """
        if self.stale:
            self._generate_batches()
        return len(self.batches)
