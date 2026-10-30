import os
import pickle
import warnings

import cudf
import anndata as ad
import numpy as np
import cupy as cp
import polars as pl
import scipy.sparse as sp
from cupyx.scipy.sparse import csr_matrix
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator
from tqdm import tqdm

# allocate maximum GPU memory
rmm.reinitialize(pool_allocator=True)
rmm.statistics.enable_statistics()
cp.cuda.set_allocator(rmm_cupy_allocator)

def read_tx_ref(
    path: str,
    min_qv: float = 20,
    min_transcripts_per_cell: int = 35,
    min_gene_occurrence_pct: float = 0.01,
) -> tuple[cudf.DataFrame, cudf.Series]:
    """
    Read lazily and filter.
    
    tx has columns:
    - cell_id: identifier for each cell
    - feature_name: identifier for each gene
    - overlaps_nucleus: 0 or 1
    - qv: transcript quality
    """

    # read filters
    tx = (
        pl.scan_parquet(path)
        .filter(
            (pl.col("qv") >= min_qv)
            & (pl.col("cell_id") != "UNASSIGNED")
            & pl.col("is_gene")
        )
        .select("cell_id", "feature_name", "overlaps_nucleus")
        .collect()
    )

    # to gpu
    tx = cudf.DataFrame.from_arrow(tx.to_arrow())

    # count gene transcripts only in cytoplasm. result is n_genes x 1
    gene_counts_cyto = tx[tx["overlaps_nucleus"] == 0].groupby("feature_name").size()
    gene_counts_nucl = tx[tx["overlaps_nucleus"] == 1].groupby("feature_name").size()
    ratio_cyto = gene_counts_cyto / (gene_counts_nucl + 1e-8)

    # filter to nuclei
    tx = tx[tx["overlaps_nucleus"] == 1]
    tx = tx.drop(columns="overlaps_nucleus")

    # remove small nuclei
    cell_counts = tx.groupby("cell_id").size()
    tx = tx[tx["cell_id"].isin(cell_counts[cell_counts >= min_transcripts_per_cell].index)]

    # remove rare genes; the kept genes are also the gene space for segmented cells
    gene_occurrence = tx.groupby("feature_name")["cell_id"].nunique()
    n_cells = tx["cell_id"].nunique()
    tx = tx[tx["feature_name"].isin(gene_occurrence[gene_occurrence >= min_gene_occurrence_pct * n_cells].index)]

    # one row per unique cell-gene pair
    tx = tx.drop_duplicates()

    return tx, ratio_cyto


def read_h5ad_ref(
    path: str,
    min_transcripts_per_cell: int = 35,
    min_gene_occurrence_pct: float = 0.01,
) -> tuple[cudf.DataFrame, cudf.Series]:
    """
    Read a cell x gene count matrix and filter like read_tx_ref.

    Returns unique cell_id / feature_name pairs; ratio_cyto is 1 for every
    gene since nuclear vs cytoplasmic counts are unknown.
    """
    adata = ad.read_h5ad(path)
    ratio_cyto = cudf.Series(np.ones(adata.n_vars), index=adata.var_names.to_numpy())

    # remove small cells
    cell_counts = np.asarray(adata.X.sum(axis=1)).ravel()
    adata = adata[cell_counts >= min_transcripts_per_cell]

    # remove rare genes
    gene_occurrence = np.asarray((adata.X > 0).sum(axis=0)).ravel()
    adata = adata[:, gene_occurrence >= min_gene_occurrence_pct * adata.n_obs]

    # one row per nonzero cell-gene pair
    X = sp.coo_matrix(adata.X)
    X.eliminate_zeros()
    tx = cudf.DataFrame({
        "cell_id": adata.obs_names.to_numpy()[X.row],
        "feature_name": adata.var_names.to_numpy()[X.col],
    })

    return tx, ratio_cyto


def compute_jaccard(
    tx: cudf.DataFrame,
    nuclear_coexpr_max: float = 0.01,
    batch_size: int = 256,
) -> dict:
    """
    Derive occurence / co-occurence matrix using sparse matrix representations
    """
    # One row per unique cell-gene pair
    pairs = tx[["cell_id", "feature_name"]].drop_duplicates()

    cell_codes, _ = pairs["cell_id"].factorize() # returns (0, 1, 2, ..), (id0, id1, id2, ...)
    gene_codes, feature_names = pairs["feature_name"].factorize()

    n_cells = int(cell_codes.max()) + 1
    n_genes = int(gene_codes.max()) + 1

    # sparse binary cell × gene matrix; 1 if gene occurs
    X = csr_matrix(
        (
            cp.ones(len(pairs), dtype=cp.float64),
            (cell_codes, gene_codes),
        ),
        shape=(n_cells, n_genes),
    )

    # occurence of each gene
    occurrence = cp.asarray(X.sum(axis=0)).ravel()

    # Gene co-occurrence counts (intersection)
    # - batch over gene blocks, including each block with itself
    # - dense blocks so pairs that never co-occur (J = 0) are kept
    values, genes_i, genes_j = [], [], []

    for i in tqdm(range(0, n_genes, batch_size), desc="Processing gene batches"):
        X_i = X[:, i:i + batch_size]
        occ_i = occurrence[i:i + batch_size]
        for j in range(i, n_genes, batch_size):
            X_j = X[:, j:j + batch_size]
            occ_j = occurrence[j:j + batch_size]

            # intersection, union and jaccard for the block
            intersection = (X_i.T @ X_j).toarray()
            union = occ_i[:, None] + occ_j[None, :] - intersection
            jacc = intersection / cp.maximum(union, 1)

            # pairs g < h of eligible genes below the nuclear co-expression cut-off
            keep = jacc < nuclear_coexpr_max
            if i == j:
                keep = cp.triu(keep, 1)
            rows, cols = cp.nonzero(keep)

            values.append(jacc[rows, cols])
            genes_i.append((rows + i).astype(cp.int32))
            genes_j.append((cols + j).astype(cp.int32))

    jaccard = {
        "values": cp.concatenate(values), # n_pairs x 1
        "genes_i": cp.concatenate(genes_i), # n_pairs x 1
        "genes_j": cp.concatenate(genes_j), # n_pairs x 1
        "occurence": occurrence, # n_genes x 1
        "feature_names": feature_names # n_genes x 1
    }
    return jaccard


def get_pairs(
    jaccard,
    k: int = 500,
) -> dict:
    idx = cp.argsort(jaccard["values"])[:k]
    genes_i = jaccard["genes_i"][idx]
    genes_j = jaccard["genes_j"][idx]
    values = jaccard["values"][idx]
    names = jaccard["feature_names"]
    return {
        "genes_idx": genes_i,
        "genes_jdx": genes_j,
        "values": values,
        "genes_i": names[genes_i],
        "genes_j": names[genes_j],
    }


def read_h5ad(
    path: str,
    min_counts: int = 35,
) -> ad.AnnData:
    """Read and filter"""
    # read
    adata = ad.read_h5ad(path)

    # count
    adata.obs.loc[:, "n_counts"] = np.array(adata.X.sum(axis=1)).ravel()
    
    # filter
    adata = adata[adata.obs["n_counts"] >= min_counts]

    # normalise
    adata.X = (adata.X.multiply(1 / adata.obs["n_counts"].values[:, None])).tocsr().copy()
    return adata


def score_pairs(
    jaccard: dict,
    adata: ad.AnnData,
    ratio_cyto: cudf.Series,
    pairs: dict,
    binarize: bool = False,
    weighted: bool = False,
    cap_zero: bool = False,
    eps: float = 1e-8,
) -> dict:
    """
    score observed co-expression
    1. get sum of min & max for each gene pair across cells
    2. get difference from "null"
    """
    # binarise
    if binarize:
        adata.X.data[:] = 1

    # gene lookups on cpu; genes without cytoplasmic counts get ratio 0
    feature_names = jaccard["feature_names"].to_numpy()
    occurrence = jaccard["occurence"].get()
    ratio = ratio_cyto.to_pandas().reindex(feature_names).fillna(0.0).to_numpy()

    genes_i = pairs["genes_idx"].get()
    genes_j = pairs["genes_jdx"].get()
    jacc_vals = pairs["values"].get()

    # every pair gene must be a column of adata
    cols = adata.var_names.get_indexer(feature_names)
    pair_genes = np.union1d(genes_i, genes_j)
    missing = feature_names[pair_genes[cols[pair_genes] < 0]]
    if len(missing) > 0:
        warnings.warn(
            f"{missing.size} pair entries involve missing genes and will be ignored; "
            f"missing genes: {list(missing[:5])}"
        )
        # remove pairs that involve missing genes
        valid_pairs = (cols[genes_i] >= 0) & (cols[genes_j] >= 0)
        genes_i, genes_j, jacc_vals = genes_i[valid_pairs], genes_j[valid_pairs], jacc_vals[valid_pairs]

    # soft jaccard for all pairs at once; csc for fast column slicing
    X = adata.X.tocsc()
    x_i = X[:, cols[genes_i]]
    x_j = X[:, cols[genes_j]]
    min_sum = x_i.minimum(x_j).sum(axis=0).A1
    max_sum = x_i.maximum(x_j).sum(axis=0).A1
    seg = min_sum / (max_sum + eps)

    # excess over nuclear baseline, weighted by occurrence and cytoplasmic penalty
    excess = seg - jacc_vals
    if cap_zero:
        excess = np.maximum(excess, 0)
    if weighted:
        weights = np.sqrt(occurrence[genes_i] * occurrence[genes_j]) / np.sqrt(
            (1 + ratio[genes_i]) * (1 + ratio[genes_j])
        )
    else:
        weights = np.ones_like(excess)
    
    # check valid
    valid = np.isfinite(excess) & np.isfinite(weights) & (weights > 0)
    if any(~valid):
        warnings.warn(f"{np.sum(~valid)} pairs are invalid and will be ignored.")
    excess, weights = excess[valid], weights[valid]

    # get score
    score = np.sum(weights * excess) / np.sum(weights)
    var = np.sum(weights * (excess - score) ** 2) / np.sum(weights)
    n_eff = np.sum(weights) ** 2 / np.sum(weights ** 2)

    # get string and values of pairs
    pairs = {
        "genes_i": feature_names[genes_i[valid]],
        "genes_j": feature_names[genes_j[valid]],
        "values": jacc_vals[valid],
        "ratio_cyto": ratio[genes_i[valid]],
        "excess": excess,
        "weights": weights,
        "binarize": binarize,
        "k_pairs": len(genes_i)
    }

    sce = {
        "score": score,
        "var": var,
        "n_eff": n_eff,
        "pairs": pairs
    }

    return sce


def spurious_coexpression(
    path_reference,
    path_query,
    path_cache: str = None,
    pop_pairs: bool = True,
    other_scores_func=None,
    **kwargs
):
    # get reference jaccard
    path_jaccard = os.path.join(path_cache, "jaccard.pkl") if path_cache else None

    if path_jaccard and os.path.exists(path_jaccard):
        print(f"Loading cached results from '{path_jaccard}'")
        with open(path_jaccard, "rb") as f:
            cache = pickle.load(f)
            jaccard = cache["jaccard"]
            ratio_cyto = cache["ratio_cyto"]
    else:
        if path_reference.endswith(".h5ad"):
            tx, ratio_cyto = read_h5ad_ref(
                path=path_reference,
                min_transcripts_per_cell=kwargs.get("min_transcripts_per_cell", 35),
                min_gene_occurrence_pct=kwargs.get("min_gene_occurrence_pct", 0.01),
            )
        else:
            tx, ratio_cyto = read_tx_ref(
                path=path_reference,
                min_qv=kwargs.get("min_qv", 20),
                min_transcripts_per_cell=kwargs.get("min_transcripts_per_cell", 35),
                min_gene_occurrence_pct=kwargs.get("min_gene_occurrence_pct", 0.01),
            )
        jaccard = compute_jaccard(tx, nuclear_coexpr_max=kwargs.get("nuclear_coexpr_max", 0.01))
        del tx

        if path_jaccard:
            os.makedirs(path_cache, exist_ok=True)
            print(f"Caching results to '{path_jaccard}'")
            with open(path_jaccard, "wb") as f:
                pickle.dump({
                    "jaccard": jaccard,
                    "ratio_cyto": ratio_cyto
                }, f)

    pairs = get_pairs(jaccard, k=kwargs.get("k", 500))


    # get scores
    if isinstance(path_query, str):
        path_query = [path_query]

    scores = []
    for i, pq in enumerate(path_query):
        print(f"--- Processing query {i+1}/{len(path_query)}: {pq} ---")
        adata = read_h5ad(
            path=pq,
            min_counts=kwargs.get("min_counts", 35),
        )

        # order imporant bc binarize is inplace
        sce_nobinary = score_pairs(
            jaccard=jaccard,
            adata=adata,
            ratio_cyto=ratio_cyto,
            pairs=pairs,
            binarize=False,
            weighted=kwargs.get("weighted", False),
            cap_zero=kwargs.get("cap_zero", False),
        )

        sce_binary = score_pairs(
            jaccard=jaccard,
            adata=adata,
            ratio_cyto=ratio_cyto,
            pairs=pairs,
            binarize=True,
            weighted=kwargs.get("weighted", False),
            cap_zero=kwargs.get("cap_zero", False),
        )

        if other_scores_func is not None:
            other_scores = other_scores_func(adata, pairs)
        else:
            other_scores = None

        # pop the pairs
        if pop_pairs:
            sce_nobinary.pop("pairs", None)
            sce_binary.pop("pairs", None)

        scores.append({
            "sce_nobinary": sce_nobinary,
            "sce_binary": sce_binary,
            "other_scores": other_scores
        })

    return scores
