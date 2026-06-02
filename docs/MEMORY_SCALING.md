# Memory scaling: VRAM-bounded gene-panel splitting

Large gene panels (e.g. 5k-gene Xenium Prime) can build a heterogeneous graph
too big to fit one GPU. Tiling caps the *per-batch* GPU footprint, but the full
`HeteroData` (all `tx` nodes + tx–tx / tx–bd edges) is assembled in host RAM
first, and that assembly — plus the per-tile work — scales with the number of
transcripts. Gene-panel splitting reduces what goes *into* the graph.

## Why transcripts, not genes

When a gene subset is selected, `ISTDataModule.load()` filters the full
transcript table (all compartments) to that subset *before* building the graph.
So peak memory scales with the **total transcript count** of the subset's
genes. Per-gene abundance is heavily skewed (a handful of genes — e.g. MALAT1 —
can dominate), so capping *genes per subset* does **not** bound memory: a subset
that happens to catch the abundant genes still OOMs.

`segger segment` therefore balances by transcript count:

- **`--max-transcripts-per-split N`** (primary): split the panel into
  `K = ceil(total_ref_transcripts / N)` subsets, each holding ≈N transcripts.
- **`--max-genes-per-split M`** (secondary): optional cap on vocabulary size per
  subset, applied on top of the transcript budget.

Genes are assigned by **greedy longest-processing-time within Phenograph
clusters**: iterating cluster-by-cluster keeps every subset panel-diverse (spans
cell types); the LPT rule (heaviest gene → currently-lightest subset) keeps the
per-subset transcript totals close. The split is deterministic given the panel
and counts. The realized per-subset transcript min/max/mean is logged so you can
tune the budget.

The split is over **disjoint** gene sets and cell ids come from the input
boundaries (shared across runs), so each transcript is produced by exactly one
subset and the final merge is a plain concat — no spatial reconciliation.

> Note: this is an *approximation* of a full-panel run (each subset's GNN sees a
> sparser tx–tx graph and subset-only cell embeddings). Validate the gap with
> `segger-0.2.0/scripts/validate_gene_split.py` before relying on it for
> production. `cells_min_counts` is forced to 0 for subset runs so cell
> membership stays identical across subsets.

## Picking the budget

Peak VRAM is roughly linear in subset transcript count: `peak ≈ a·n_tx + b`.
Calibrate once: run a single small subset, watch `nvidia-smi` (or the memory
tracker on `optimisation/quadtree_memory_optimisation`), then scale linearly to
your GPU. For a ~50 GB Athera GPU, start conservative and raise `N` until a
subset approaches the limit.

## Running it

Single machine / one GPU (runs subsets sequentially in-process):

```bash
segger segment -i DATA -o OUT --max-transcripts-per-split 40000000
```

Three composable steps (for an HPC DAG):

```bash
segger segment   -i DATA -o OUT --max-transcripts-per-split 40000000 --plan-only   # writes OUT/gene_split_plan.parquet
segger segment   -i DATA -o OUT --split-plan OUT/gene_split_plan.parquet --subset-id K  # one subset (per array task)
segger merge-splits -o OUT                                                          # concat -> OUT/segger_segmentation.parquet
```

### Athera (LSF)

`segger-0.2.0/scripts/submit_gene_split_lsf.sh` wires these into a DAG: a
blocking **plan** job, a **subset array** (`seg[1-K]`, one GPU each, bounded by
the plan), then a **merge** job. Resumable — re-submitting skips subsets whose
parquet already exists. Set `PYTHONNOUSERSITE=1` and a local `TMPDIR` to avoid
`~/.local` shadowing and NFS `.nfs*` cleanup errors.

## Hardware notes

- **Mac:** segger does **not** import on macOS — `data/utils/neighbors.py` and
  `anndata.py` require RAPIDS (`cuml`/`cudf`/`cugraph`/`cupy`), which is
  CUDA/NVIDIA-only with no macOS build. (The pure split helpers in
  `data/utils/gene_split.py` import without the GPU stack, which is why the
  split unit tests can be reasoned about off-GPU, but a full run cannot.)
- **Google Colab TPU:** not supported (RAPIDS is CUDA-only; PyG scatter ops are
  not XLA-friendly).
- **Google Colab GPU:** supported and a good fit. On a 16 GB T4, set
  `--max-transcripts-per-split` to fit 16 GB and run subsets sequentially — the
  cheapest place to validate the split before moving to Athera.
