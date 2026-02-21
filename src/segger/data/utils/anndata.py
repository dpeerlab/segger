from __future__ import annotations

from typing import TYPE_CHECKING
from torch.nn.functional import normalize
from scipy import sparse as sp
import polars as pl
import pandas as pd
import numpy as np
import torch
import os

def _lazy_imports():
    global gpd, sc, sklearn, cupyx, cuml
    import geopandas as gpd
    import scanpy as sc
    import sklearn
    import cupyx
    import cuml

if TYPE_CHECKING:  # pragma: no cover
    import geopandas as gpd

from ...io.fields import TrainingTranscriptFields, TrainingBoundaryFields
from .neighbors import phenograph_rapids


def _debug_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


_UNASSIGNED_CELL_ID_MARKERS = {
    "",
    "UNASSIGNED",
    "NONE",
    "NULL",
    "NAN",
    "NA",
    "-1",
}


def _assigned_cell_mask(cell_id_column: str) -> pl.Expr:
    """Mask for transcripts with valid assigned cell IDs."""
    cell_id_str = (
        pl.col(cell_id_column)
        .cast(pl.Utf8)
        .fill_null("")
        .str.strip_chars()
        .str.to_uppercase()
    )
    return pl.col(cell_id_column).is_not_null() & (~cell_id_str.is_in(_UNASSIGNED_CELL_ID_MARKERS))


def _to_int_array(values: pd.Series | np.ndarray, missing: int = -1) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").fillna(missing)
    return numeric.astype(np.int64).to_numpy()


def _print_clustering_diagnostics(
    ad,
    cells_min_counts: int,
    cells_clusters_n_neighbors: int,
    cells_clusters_resolution: float,
    genes_min_counts: int,
    genes_clusters_n_neighbors: int,
    genes_clusters_resolution: float,
) -> None:
    """Print concise diagnostics for AnnData clustering outputs."""
    def _safe_sim_stats(values: np.ndarray) -> tuple[tuple[int, ...], float, float]:
        arr = np.asarray(values)
        if arr.size == 0:
            return tuple(arr.shape), float("nan"), float("nan")
        try:
            return tuple(arr.shape), float(np.nanmin(arr)), float(np.nanmax(arr))
        except Exception:
            return tuple(arr.shape), float("nan"), float("nan")

    filtered = ad.obs["filtered"].to_numpy(dtype=bool)
    cell_clusters = _to_int_array(ad.obs["phenograph_cluster"], missing=-1)
    clustered_cells = cell_clusters >= 0
    filtered_clustered = filtered & clustered_cells
    cluster_sizes = pd.Series(cell_clusters[filtered_clustered]).value_counts().to_numpy()

    gene_clusters = _to_int_array(ad.var["phenograph_cluster"], missing=-1)
    valid_gene_clusters = gene_clusters[gene_clusters >= 0]
    gene_cluster_sizes = pd.Series(valid_gene_clusters).value_counts().to_numpy()

    cell_sim_shape, cell_sim_min, cell_sim_max = _safe_sim_stats(
        ad.uns.get("cell_cluster_similarities")
    )
    gene_sim_shape, gene_sim_min, gene_sim_max = _safe_sim_stats(
        ad.uns.get("gene_cluster_similarities")
    )

    print(
        "[segger][diag][cluster] "
        f"cells_total={ad.n_obs}, cells_filtered={int(filtered.sum())}, "
        f"cells_clustered={int(filtered_clustered.sum())}, "
        f"cells_unclustered_filtered={int((filtered & ~clustered_cells).sum())}",
        flush=True,
    )
    print(
        "[segger][diag][cluster] "
        f"cell_clusters={int(np.unique(cell_clusters[clustered_cells]).size)}, "
        f"cell_cluster_size_min={int(cluster_sizes.min()) if cluster_sizes.size else 0}, "
        f"cell_cluster_size_med={float(np.median(cluster_sizes)) if cluster_sizes.size else 0.0:.1f}, "
        f"cell_cluster_size_max={int(cluster_sizes.max()) if cluster_sizes.size else 0}",
        flush=True,
    )
    print(
        "[segger][diag][cluster] "
        f"genes_total={ad.n_vars}, genes_clusters={int(np.unique(valid_gene_clusters).size)}, "
        f"gene_cluster_size_min={int(gene_cluster_sizes.min()) if gene_cluster_sizes.size else 0}, "
        f"gene_cluster_size_med={float(np.median(gene_cluster_sizes)) if gene_cluster_sizes.size else 0.0:.1f}, "
        f"gene_cluster_size_max={int(gene_cluster_sizes.max()) if gene_cluster_sizes.size else 0}",
        flush=True,
    )
    print(
        "[segger][diag][cluster] "
        f"cell_similarity_shape={cell_sim_shape}, "
        f"cell_similarity_range=({cell_sim_min:.4f}, {cell_sim_max:.4f}), "
        f"gene_similarity_shape={gene_sim_shape}, "
        f"gene_similarity_range=({gene_sim_min:.4f}, {gene_sim_max:.4f})",
        flush=True,
    )
    print(
        "[segger][diag][cluster] "
        f"params: cells_min_counts={cells_min_counts}, "
        f"cells_neighbors={cells_clusters_n_neighbors}, "
        f"cells_resolution={cells_clusters_resolution}, "
        f"cells_min_cluster_size=100, "
        f"genes_min_counts={genes_min_counts}, "
        f"genes_neighbors={genes_clusters_n_neighbors}, "
        f"genes_resolution={genes_clusters_resolution}, "
        f"genes_min_cluster_size=-1",
        flush=True,
    )


def anndata_from_transcripts(
    tx: pl.DataFrame,
    feature_column: str,
    cell_id_column: str,
    score_column: str | None = None,
    coordinate_columns: list[str] | None = None,
    feature_vocab: list[str] | None = None,
):
    """TODO: Add description.
    """
    _lazy_imports()
    # Keep only transcripts with valid assigned cell IDs.
    tx = tx.filter(_assigned_cell_mask(cell_id_column))
    # Get sparse counts from transcripts
    if feature_vocab is None:
        feature_idx = tx.select(
            feature_column).unique().with_row_index()
    else:
        feature_vocab = [str(gene) for gene in feature_vocab]
        if len(feature_vocab) != len(set(feature_vocab)):
            raise ValueError(
                "feature_vocab contains duplicate genes. "
                "Gene vocabulary must be unique to preserve checkpoint mapping."
            )
        feature_idx = pl.DataFrame(
            {feature_column: feature_vocab}
        ).with_row_index()
        tx = tx.filter(pl.col(feature_column).is_in(feature_vocab))

    segment_idx = tx.select(
        cell_id_column).unique().with_row_index()
    groupby = (
        tx
        .with_columns(
            # Map feature to numeric id
            pl.col(feature_column)
            .replace_strict(
                old=feature_idx[feature_column],
                new=feature_idx["index"],
                return_dtype=pl.UInt32,
            )
            .alias('_fid'),
            # Map segmentation to numeric id
            pl.col(cell_id_column)
            .replace_strict(
                old=segment_idx[cell_id_column],
                new=segment_idx["index"],
                return_dtype=pl.UInt32,
            )
            .alias('_sid'),
        )
        # Create sparse count matrix
        .group_by(['_sid', '_fid'])
    )
    # Get correlation matrix
    ijv = groupby.len().to_numpy().T
    X = sp.coo_matrix(
        (ijv[2], ijv[:2]),
        shape=(len(segment_idx), len(feature_idx)),
    ).tocsr()
    
    # To AnnData
    adata = sc.AnnData(
        X=X,
        obs=pd.DataFrame(
            index=(
                segment_idx
                .get_column(cell_id_column)
                .to_numpy()
                .astype(str)
            )
        ),
        var=pd.DataFrame(
            index=(
                feature_idx
                .get_column(feature_column)
                .to_numpy()
                .astype(str)
            )
        ),
    )
    # Optionally: Add transcript scores
    if score_column is not None:
        ijv = groupby.agg(pl.col(score_column).mean()).to_numpy().T
        adata.layers[f'{score_column}_scores'] = sp.coo_matrix(
            (ijv[2], ijv[:2].astype(int))).tocsr()

    # Optionally: Add coordinates
    if coordinate_columns is not None:
        centroids = (
            tx
            .group_by(cell_id_column)
            .agg([pl.col(c).mean().alias(c) for c in coordinate_columns])
        )
        coords = (
            centroids
            .to_pandas()
            .set_index(cell_id_column)
            .loc[adata.obs.index, coordinate_columns]
        )
        adata.obsm["X_spatial"] = coords.values

    return adata


def get_cluster_cosine_similarity(
    embedding: torch.Tensor,
    clusters: torch.Tensor,
) -> torch.Tensor:
    """TODO: Add description.
    """
    # Get label mapping
    unique, inverse = clusters.unique(sorted=False, return_inverse=True)
    
    # Empty output tensor
    k = unique.numel()
    sums = torch.zeros(
        k,
        embedding.size(1),
        dtype=embedding.dtype,
        device=embedding.device,
    )
    # Compute average cosine distance
    embedding = normalize(embedding, p=2, dim=1, eps=1e-8)
    sums.index_add_(0, inverse, embedding)
    counts = torch.bincount(inverse, minlength=k).unsqueeze(1)
    means = sums / counts

    return means @ means.T


def setup_anndata(
    transcripts: pl.DataFrame,
    boundaries: gpd.GeoDataFrame,
    cell_column: str,
    cells_embedding_size: int,
    cells_min_counts: int,
    cells_clusters_n_neighbors: int,
    cells_clusters_resolution: float,
    genes_min_counts: int,
    genes_clusters_n_neighbors: int,
    genes_clusters_resolution: float,
    compute_morphology: bool = False,
    feature_vocab: list[str] | None = None,
):
    """TODO: Add description.
    """
    _lazy_imports()
    if feature_vocab is not None:
        feature_vocab = [str(gene) for gene in feature_vocab]
    # Standard fields
    tx_fields = TrainingTranscriptFields()
    bd_fields = TrainingBoundaryFields()

    # Build AnnData from transcript counts
    ad = anndata_from_transcripts(
        transcripts,
        tx_fields.feature,
        cell_column,
        coordinate_columns=[tx_fields.x, tx_fields.y],
        feature_vocab=feature_vocab,
    )
    if ad.n_obs == 0:
        raise ValueError(
            "No transcripts with valid cell assignments were found for AnnData construction."
        )
    if ad.n_vars == 0:
        raise ValueError(
            "No genes available to build AnnData. "
            "Check input filtering and checkpoint vocabulary overlap."
        )

    # Map boundary cell IDs to boundary index
    ad.obs = (
        ad.obs
        .join(
            (
                boundaries
                .reset_index(names=bd_fields.index)
                .set_index(bd_fields.id, verify_integrity=True)
                .get(bd_fields.index)
            ),
            how="left",
            validate="1:1",
        )
        .reset_index(names=bd_fields.id)
        .set_index(bd_fields.index, verify_integrity=True)
    )
    assert ~ad.obs.index.isna().any()

    # Remove low-count genes unless a fixed checkpoint vocabulary is provided
    ad.var['n_counts'] = ad.X.sum(0).A.flatten()
    if feature_vocab is None:
        ad = ad[:, ad.var['n_counts'].ge(genes_min_counts)]

    # Explicitly sort indices for reproducibility unless vocab order is fixed
    if feature_vocab is None:
        ad = ad[ad.obs.index.sort_values(), ad.var.index.sort_values()]
    else:
        ad = ad[ad.obs.index.sort_values(), feature_vocab]
    
    # Add raw counts
    ad.raw = ad.copy()
    ad.layers['counts'] = ad.raw.X.copy()

    # Keep track of filtered cells
    ad.obs['n_counts'] = ad.raw.X.sum(1).A.flatten()
    ad.obs['filtered'] = ad.obs['n_counts'].ge(cells_min_counts)

    # Normalize to filtered dataset counts
    ad.layers['norm'] = ad.layers['counts'].copy()
    target_sum = ad.obs.loc[ad.obs['filtered'], 'n_counts'].median()
    sc.pp.normalize_total(ad, target_sum=target_sum, layer='norm')

    # Build gene embedding on filtered dataset
    C = np.corrcoef(ad[ad.obs['filtered']].layers['norm'].todense().T)
    C = np.nan_to_num(C, 0, posinf=True, neginf=True)
    model = sklearn.decomposition.PCA(n_components=cells_embedding_size)
    ad.varm['X_corr'] = model.fit_transform(C)

    # Build PCs on filtered cells and project all cells
    counts_sparse_gpu = cupyx.scipy.sparse.csr_matrix(ad.layers['norm'])
    model = cuml.PCA(n_components=cells_embedding_size)
    model.fit(counts_sparse_gpu[ad.obs['filtered'].values])
    ad.obsm['X_pca'] = model.transform(counts_sparse_gpu).get()

    # Compute clusters on filtered cells
    cell_clusters = phenograph_rapids(
        ad[ad.obs['filtered']].obsm['X_pca'],
        n_neighbors=cells_clusters_n_neighbors, 
        resolution=cells_clusters_resolution,
        min_size=100,
    )
    ad.obs['phenograph_cluster'] = -1  # removed cells have no cluster
    ad.obs.loc[ad.obs['filtered'], 'phenograph_cluster'] = cell_clusters
    ad.obs['phenograph_cluster'] = pd.Categorical(ad.obs['phenograph_cluster'])

    # Compute pairwise cosine similarities among cell clusters
    ad.uns['cell_cluster_similarities'] = get_cluster_cosine_similarity(
        embedding=torch.tensor(ad.obsm['X_pca']),
        clusters=torch.tensor(ad.obs['phenograph_cluster'].values),
    ).numpy()

    # Compute clusters on genes from embedding
    ad.var['phenograph_cluster'] = phenograph_rapids(
        ad.varm['X_corr'],
        n_neighbors=genes_clusters_n_neighbors,
        resolution=genes_clusters_resolution,
        min_size=-1,
    )
    ad.var['phenograph_cluster'] = pd.Categorical(ad.var['phenograph_cluster'])

    # Compute pairwise cosine similarities among gene clusters
    ad.uns['gene_cluster_similarities'] = get_cluster_cosine_similarity(
        embedding=torch.tensor(ad.varm['X_corr']),
        clusters=torch.tensor(ad.var['phenograph_cluster'].values),
    ).numpy()
    # Add cell and gene numeric encodings to AnnData
    ad.obs[tx_fields.cell_encoding] = np.arange(len(ad.obs)).astype(int)
    ad.var[tx_fields.gene_encoding] = np.arange(len(ad.var)).astype(int)

    if _debug_flag("SEGGER_DEBUG_CLUSTERING"):
        _print_clustering_diagnostics(
            ad=ad,
            cells_min_counts=cells_min_counts,
            cells_clusters_n_neighbors=cells_clusters_n_neighbors,
            cells_clusters_resolution=cells_clusters_resolution,
            genes_min_counts=genes_min_counts,
            genes_clusters_n_neighbors=genes_clusters_n_neighbors,
            genes_clusters_resolution=genes_clusters_resolution,
        )

    if compute_morphology:
        from segger.geometry.morphology import get_polygon_props
        # # make sure index matches by cell_id
        boundaries = boundaries.set_index(bd_fields.id, verify_integrity=True)
        boundaries = boundaries.loc[ad.obs[bd_fields.id]]
        # Compute morphology properties and add to AnnData
        morpho_props = get_polygon_props(
            boundaries.geometry,
            area=True,
            convexity=True,
            elongation=True,
            circularity=True,
        )
        for col in morpho_props.columns:
            ad.obs[col] = morpho_props[col].values
        # concat all morphology properties into a single embedding
        ad.obsm['X_morphology'] = morpho_props.to_numpy(dtype=np.float32)
    return ad
