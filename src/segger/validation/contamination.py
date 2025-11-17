from __future__ import annotations

from typing import Dict, List

import cupy as cp
import scanpy as sc
import cuml
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp
from anndata import AnnData
from scipy import sparse


def map_with_default(
    keys: List[str] | np.ndarray,
    mapping: Dict[str, int],
    default: int = -1,
    dtype: np.dtype = np.int32,
) -> np.ndarray:
    """Return an array of integer ids for *keys*.

    Parameters
    ----------
    keys : list[str] or ndarray
        Keys to look up.
    mapping : dict[str, int]
        Mapping from key to id.
    default : int, optional
        Id to use when *key* is absent. Default "-1".
    dtype : numpy.dtype, optional
        Output dtype. Default "np.int32".
    """
    out = np.empty(len(keys), dtype=dtype)
    for i, k in enumerate(keys):
        out[i] = mapping.get(str(k), default)
    return out

def get_neighbor_frequencies(
    ad: AnnData,
    k: int,
    col: str,
    obsm: str = "X_spatial",
    normalize: bool = True,
    key_added: str = "neighbor_frequencies",
    max_distance: float | None = None,
) -> pd.DataFrame:
    """Compute neighbour cell-type frequencies per cell.

    Parameters
    ----------
    ad
        AnnData with coordinates in ``ad.obsm[obsm]``.
    k
        Number of nearest neighbours queried for each cell.
    col
        Column in ``ad.obs`` containing cell-type labels.
    obsm
        Key in ``.obsm`` with coordinate matrix. Default ``"X_spatial"``.
    normalize
        If *True* rows are normalised to sum to 1. Default *True*.
    key_added
        Key under which the DataFrame is stored in ``ad.obsm``.
    max_distance
        Optional distance threshold. Neighbours farther than this value
        are ignored when building the frequency table.

    Returns
    -------
    pandas.DataFrame
        Cell x type frequency table (rows may sum to <1 when filtering).
    """
    X = cp.asarray(ad.obsm[obsm])
    nn = cuml.neighbors.NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    dist_cu, idx_cu = nn.kneighbors(X)

    labels, cell_types = pd.factorize(ad.obs[col], sort=True)
    n_types = len(cell_types)

    host = np.repeat(np.arange(ad.n_obs, dtype=np.int32), k)
    neigh = idx_cu.flatten().get()
    dists = dist_cu.flatten().get()
    if max_distance is not None:
        mask = dists <= max_distance
        host = host[mask]
        neigh = neigh[mask]
    cols_ = labels[neigh].astype(np.int32)
    data = np.ones_like(cols_, dtype=np.int32)

    mat = sp.csr_matrix((data, (host, cols_)), shape=(ad.n_obs, n_types))
    if normalize:
        sums = np.asarray(mat.sum(1)).ravel()
        sums[sums == 0] = 1.0
        mat = mat.multiply(1.0 / sums[:, None])

    df = pd.DataFrame(mat.toarray(), index=ad.obs_names, columns=cell_types)
    ad.obsm[key_added] = df
    return df

def calculate_contamination(
    adata: AnnData,
    reference: pl.DataFrame,
    *,
    counts_layer: str,
    spatial_key: str,
    cell_type_key: str,
    n_neighbors: int = 10,
    max_neighbor_distance: float = 20,
    alpha_self: float = 0.8,
    alpha_neighbor: float = 0.15,
    alpha_background: float = 0.05,
    reference_cell_type_key: str = "cell_type_name",
    reference_gene_name_key: str = "gene_name",
    eps: float = 1e-6,
    contam_cutoff: float = 0.5,
) -> None:
    """add probability layers and contamination percentage to *adata*.

    The function creates three CSR layers (*q_self*, *q_neighbor*,
    *q_background*) and a new column "percent_contamination".
    """
    # neighbour frequencies per cell
    tmp_key = "neighbor_frequencies"
    get_neighbor_frequencies(
        adata,
        k=n_neighbors,
        max_distance=max_neighbor_distance,
        col=cell_type_key,
        obsm=spatial_key,
        normalize=True,
        key_added=tmp_key,
    )
    neigh_df: pd.DataFrame = adata.obsm[tmp_key]

    # reference likelihood matrix L[type, gene]
    ct_map = {ct: i for i, ct in enumerate(sorted(reference[reference_cell_type_key].unique()))}
    gn_map = {g: i for i, g in enumerate(sorted(reference[reference_gene_name_key].unique()))}

    n_types = len(ct_map)
    n_genes = len(gn_map)

    L = np.full((n_types, n_genes), eps, dtype=np.float32)
    for row in reference.iter_rows(named=True):
        ct_id = ct_map[row[reference_cell_type_key]]
        g_id = gn_map[row[reference_gene_name_key]]
        L[ct_id, g_id] = row.get("pc", 1.0) * row.get("me", 1.0) + eps

    neigh_df = neigh_df.reindex(columns=list(ct_map.keys()), fill_value=0.0)
    neigh = neigh_df.to_numpy(dtype=np.float32)

    A = adata.obs[cell_type_key].value_counts(normalize=True)
    A = A.reindex(ct_map.keys(), fill_value=0.0).to_numpy()

    # sparse counts indices
    X_layer = adata.layers[counts_layer]
    X = X_layer.tocoo() if isinstance(X_layer, sp.spmatrix) else X_layer.to_coo()
    rows, cols, vals = X.row, X.col, X.data
    
    host_ct_idx_all = map_with_default(
        adata.obs[cell_type_key].astype(str), ct_map, -1
    )
    host_ct_idx = host_ct_idx_all[rows]
    gene_idx_all = map_with_default(adata.var_names, gn_map, -1)
    gene_idx = gene_idx_all[cols]
    missing_gene = gene_idx == -1

    # likelihoods per transcript
    P_self = np.where(missing_gene, eps, L[host_ct_idx, gene_idx])

    nv = neigh[rows].copy()
    mask_valid = (~missing_gene) & (host_ct_idx >= 0)
    idx_valid = np.nonzero(mask_valid)[0]
    if idx_valid.size > 0:
        nv[idx_valid, host_ct_idx[idx_valid]] = 0.0
    P_neigh = (nv * L[:, gene_idx].T).sum(axis=1) + eps

    P_back = A @ L[:, gene_idx] + eps

    # weight with alphas and normalise
    q_self = alpha_self * P_self
    q_neigh = alpha_neighbor * P_neigh
    q_back = alpha_background * P_back
    denom = q_self + q_neigh + q_back
    q_self /= denom
    q_neigh /= denom
    q_back /= denom

    q_self[missing_gene] = 0
    q_neigh[missing_gene] = 0
    q_back[missing_gene] = 0

    shape = adata.layers[counts_layer].shape
    adata.layers["q_self"] = sparse.coo_matrix(
        (q_self, (rows, cols)), shape=shape
    ).tocsr()
    adata.layers["q_neighbor"] = sparse.coo_matrix(
        (q_neigh, (rows, cols)), shape=shape
    ).tocsr()
    adata.layers["q_background"] = sparse.coo_matrix(
        (q_back, (rows, cols)), shape=shape
    ).tocsr()

    # percent contamination per cell
    contam_mask = q_self < contam_cutoff
    contam_mask[missing_gene] = False
    contam_vals = np.where(contam_mask, vals, 0.0)
    adata.layers["contamination"] = sparse.coo_matrix(
        (contam_vals, (rows, cols)), shape=shape
    ).tocsr()

    contam_counts = np.bincount(
        rows[contam_mask], weights=vals[contam_mask], minlength=adata.n_obs
    )
    total_counts = np.bincount(rows, weights=vals, minlength=adata.n_obs)
    adata.obs["percent_contamination"] = (
        100.0 * contam_counts / np.maximum(total_counts, 1)
    )

def contamination_flow(
    ad: AnnData,
    reference: pl.DataFrame,
    *,
    cell_type_key: str,
    counts_layer: str,
    contamination_layer: str = "contamination",
    reference_cell_type_key: str = "cell_type_name",
    reference_gene_name_key: str = "gene_name",
) -> pd.DataFrame:
    """Donor - host contamination counts.

    Each gene's contribution is split across donor cell types according
    to reference weights (pct x mean, row-normalised). The contamination
    layer is assumed to hold counts flagged as noise.
    """
    if contamination_layer not in ad.layers:
        raise ValueError("contamination layer missing in AnnData")

    donor_types = reference[reference_cell_type_key].unique()
    genes_ref = reference[reference_gene_name_key].unique()
    d_map = {ct: i for i, ct in enumerate(donor_types)}
    g_map = {g: i for i, g in enumerate(genes_ref)}

    # weight matrix W[gene, donor] row-normalised
    W = np.zeros((len(genes_ref), len(donor_types)), dtype=np.float32)
    for row in reference.iter_rows(named=True):
        d = d_map[row[reference_cell_type_key]]
        g = g_map[row[reference_gene_name_key]]
        W[g, d] = row.get("pc", 0.0) * row.get("me", 0.0)
    row_sum = W.sum(1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    W /= row_sum

    # align genes
    gene_idx_ad = map_with_default(ad.var_names, g_map, -1)
    keep_gene = gene_idx_ad >= 0
    if not np.any(keep_gene):
        raise ValueError("No shared genes between AnnData and reference")

    C = ad.layers[contamination_layer].tocsr()[:, keep_gene]
    W_sub = W[gene_idx_ad[keep_gene], :]
    contrib = C @ W_sub  # cell - donor counts

    # percent per cell (normalised by library size)
    libsize = ad.layers[counts_layer].sum(1).A1.astype(np.float32)
    libsize[libsize == 0] = 1.0
    percent = 100.0 * (contrib / libsize[:, None])

    # aggregate by host cell type (mean percent)
    host_lab = ad.obs[cell_type_key].astype(str)
    host_types = host_lab.unique()
    h_map = {ct: i for i, ct in enumerate(host_types)}
    host_idx = host_lab.map(h_map).to_numpy()

    flow = np.zeros((len(donor_types), len(host_types)), dtype=np.float64)
    cell_counts = np.bincount(host_idx, minlength=len(host_types))
    for d in range(len(donor_types)):
        sums = np.bincount(
            host_idx,
            weights=percent[:, d],
            minlength=len(host_types),
        )
        flow[d] = sums / np.maximum(cell_counts, 1)

    flow = pd.DataFrame(flow, index=donor_types, columns=host_types)
    flow.index.name = 'source'
    flow.columns.name = 'host'

    return flow


def group_reference(
    reference: pl.DataFrame,
    grouping: Dict[str, str],
    *,
    cell_type_name_col: str = "cell_type_name",
    gene_name_col: str = "gene_name",
    percent_col: str = "pc",
    mean_expr_col: str = "me",
    n_cells_col: str = "n_cells_cell_type",
    n_pos_cells_col: str = "n",
) -> pl.DataFrame:
    """Aggregate reference rows into user-defined cell-type groups.

    Parameters
    ----------
    reference
        Polars DataFrame with per-(cell-type, gene) statistics.
    grouping
        Mapping from original cell-type names to group names.
    cell_type_name_col, gene_name_col
        Column names in *reference* identifying cell type and gene.
    percent_col, mean_expr_col, n_cells_col, n_pos_cells_col
        Column names with percent positive, mean expression, cell counts,
        and positive cell counts, respectively.

    Returns
    -------
    pl.DataFrame
        Reference aggregated by *grouping*, updating percent positive,
        mean expression, and cell counts.
    """
    ref = reference.with_columns(
        pl.col(cell_type_name_col)
        .map_elements(lambda x: grouping.get(x, x), return_dtype=pl.String)
        .alias(cell_type_name_col)
    )

    ref = ref.with_columns(
        (
            pl.col(mean_expr_col) * pl.col(n_pos_cells_col)
        ).alias("weighted_expr")
    )

    agg = (
        ref.group_by([cell_type_name_col, gene_name_col])
        .agg(
            pl.sum(n_cells_col).alias(n_cells_col),
            pl.sum(n_pos_cells_col).alias(n_pos_cells_col),
            pl.sum("weighted_expr").alias("expr_sum"),
        )
        .with_columns(
            (
                pl.col("expr_sum") / pl.col(n_pos_cells_col)
            ).fill_null(0).alias(mean_expr_col),
            (
                pl.col(n_pos_cells_col) / pl.col(n_cells_col)
            ).fill_null(0).alias(percent_col),
        )
        .drop("expr_sum")
    )
    return agg

def expression_summary_from_anndata(
    ad: sc.AnnData,
    cell_type_col: str,
    raw_layer: str,
    min_counts: int = 2
) -> pl.DataFrame:
    # TODO: Add documentation
    
    # Normalize as in CellxGene
    ad.layers['_cxg_norm'] = ad.layers[raw_layer].copy()
    sc.pp.normalize_total(ad, target_sum=1e4, layer='_cxg_norm')
    sc.pp.log1p(ad, layer='_cxg_norm')

    # Filter as in CellxGene
    mask = ad.layers[raw_layer] >= min_counts
    ad.layers['_cxg_norm'] = ad.layers['_cxg_norm'].multiply(mask)
    ad.layers['_cxg_norm'].eliminate_zeros()

    # Summary data from CellxGene expression summary
    aggs = {
        'n': 'count_nonzero',   # 1) Non-zero counts per cell type
        'me': 'sum',            # 2) Mean expression in positive cells
    }
    stats = dict()
    for name, func in aggs.items():
        stats[name] = pl.from_pandas(
            sc.get.aggregate(ad, by=cell_type_col, func=func, layer='_cxg_norm')
            .to_df(layer=func)
            .melt(value_name=name, ignore_index=False, var_name='gene_name')
            .reset_index(names='cell_type_name')
        )

    # 3) Number of cells per cell type
    n_ct = pl.from_pandas(
        ad.obs.value_counts(cell_type_col)
        .reset_index()
        .rename(
            {cell_type_col: 'cell_type_name', 'count': 'n_cells_cell_type'},
            axis=1
        )
    ).with_columns(pl.col('cell_type_name').cast(pl.String))

    # Join into summary dataframe
    summary = (
        stats['n']
        .join(stats['me'], on=['cell_type_name', 'gene_name'])
        .join(n_ct, on='cell_type_name')
        .filter(pl.col('n') > 0)
        .with_columns(pl.col('me') / pl.col('n'))
        .with_columns(pc=pl.col('n') / pl.col('n_cells_cell_type'))
        .with_columns(pl.col('n').cast(pl.Int64))
    )

    return summary