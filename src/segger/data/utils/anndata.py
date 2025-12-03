from torch.nn.functional import normalize
from scipy import sparse as sp
import geopandas as gpd
import polars as pl
import pandas as pd
import scanpy as sc
import numpy as np
import sklearn
import torch
import cupyx
import cuml

from ...io.fields import TrainingTranscriptFields, TrainingBoundaryFields
from .neighbors import phenograph_rapids
from sieve.geometry.morphology import get_polygon_props

def anndata_from_transcripts(
    tx: pl.DataFrame,
    feature_column: str,
    cell_id_column: str,
    score_column: str | None = None,
    coordinate_columns: list[str] | None = None,
):
    """TODO: Add description.
    """
    # Remove non-nuclear transcript
    tx = tx.filter(pl.col(cell_id_column).is_not_null())
    # Get sparse counts from transcripts
    feature_idx = tx.select(
        feature_column).unique().with_row_index()
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
    X = sp.coo_matrix((ijv[2], ijv[:2])).tocsr()
    
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
):
    """TODO: Add description.
    """
    # Standard fields
    tx_fields = TrainingTranscriptFields()
    bd_fields = TrainingBoundaryFields()

    # Build AnnData from transcript counts
    ad = anndata_from_transcripts(
        transcripts,
        tx_fields.feature,
        cell_column,
        coordinate_columns=[tx_fields.x, tx_fields.y],
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

    # Remove genes with fewer than min counts permanently
    ad.var['n_counts'] = ad.X.sum(0).A.flatten()
    ad = ad[:,  ad.var['n_counts'].ge(genes_min_counts)]

    # Explicitly sort indices for reproducibility
    ad = ad[ad.obs.index.sort_values(), ad.var.index.sort_values()]
    
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

    if compute_morphology:
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
