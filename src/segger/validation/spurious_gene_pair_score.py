from shapely.geometry import Polygon
import scanpy as sc
import numpy as np
import pandas as pd
import geopandas as gpd
import itertools
from scipy.stats import multinomial
import os
from pathlib import Path
from scipy.sparse import csr_matrix
from loguru import logger


def _normalize_to_median(adata: sc.AnnData) -> pd.DataFrame:
    """ Normalize counts in AnnData to median library size using multinomial sampling."""
    logger.info("Normalizing to median library size using multinomial sampling.")
    transcripts = pd.DataFrame(adata.X.toarray(), columns=adata.var_names, index=adata.obs_names)
    transcripts["sum"] = transcripts.sum(axis=1)
    median = transcripts["sum"].median()
    probabilities = transcripts.div(transcripts["sum"], axis=0).drop(columns=["sum"])
    transcripts = transcripts.drop(columns=["sum"])

    probabilities_normalized_np = probabilities.div(probabilities.sum(axis=1), axis=0).to_numpy()
    normalized_counts = np.zeros_like(probabilities_normalized_np, dtype=np.int32)

    num_cells = probabilities_normalized_np.shape[0]
    for i in range(num_cells):
        p_normalized = probabilities_normalized_np[i, :]
        p_normalized = p_normalized.astype(np.float64)
        p_normalized /= p_normalized.sum()

        if np.isnan(p_normalized).any():
            continue
        new_counts = multinomial.rvs(median, p_normalized, size=1)[0]
        
        normalized_counts[i, :] = new_counts
    normalized_df = pd.DataFrame(
        normalized_counts, 
        columns=transcripts.columns, 
        index=transcripts.index
    )
    logger.info("Normalization complete.")
    return normalized_df

def find_conditional_coexpression(adata: sc.AnnData) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    """ Find conditional co-expression scores for gene pairs."""
    logger.info("Finding conditional co-expression scores.")
    normalized_df = _normalize_to_median(adata)

    cells_expressing_genes = {
        gene  : set(normalized_df.index[normalized_df[gene] > 0]) for gene in normalized_df.columns
    }

    conditional_coexpression = pd.DataFrame(index=cells_expressing_genes.keys(), columns=cells_expressing_genes.keys())

    for i, j in itertools.permutations(cells_expressing_genes.keys(), 2):
        intersection = cells_expressing_genes[i].intersection(cells_expressing_genes[j])
        score = len(intersection) / len(cells_expressing_genes[j])
        conditional_coexpression.loc[i, j] = score
    logger.info("Conditional co-expression scores computed.")
    return conditional_coexpression


def get_expansion_conditional_coexpression(transcripts: pd.DataFrame, nucleus_gdf: gpd.GeoDataFrame, buffer_size: float | pd.Series, genes: list[str] = None) -> pd.DataFrame:
    """ Get transcripts located within the buffer regions."""
    logger.info("Computing expansion conditional co-expression.")
    nucleus_buffered_gdf = nucleus_gdf.copy()
    nucleus_buffered_gdf["geometry"] = nucleus_buffered_gdf.buffer(distance = buffer_size)
    transcripts = transcripts.copy()
    
    if genes is not None:
        transcripts = transcripts.loc[transcripts["feature_name"].isin(genes)]

    transcripts_geometry = gpd.points_from_xy(
        x=transcripts['x_location'],
        y=transcripts['y_location']
    )

    transcripts_gdf = gpd.GeoDataFrame(
        transcripts.assign(geometry=transcripts_geometry),
        geometry='geometry',
        crs=nucleus_buffered_gdf.crs
    )

    logger.info("Performing spatial join between transcripts and buffered nucleus boundaries.")

    intersected_gdf = transcripts_gdf.sjoin(
        nucleus_buffered_gdf,
        how='inner',
        predicate='intersects'
    )

    logger.info("Spatial join complete. Normalizing transcript counts within buffered regions.")

    index_name = nucleus_buffered_gdf.index.name
    if index_name is None:
        index_name = 'index'

    intersected_gdf = intersected_gdf.rename(columns={
        f'{index_name}_right': 'nucleus_id_new',
        f'{index_name}_left': 'nucleus_id_old'
    })

    intersected_gdf['feature_name'] = intersected_gdf['feature_name'].astype('category')

    def normalize_nucleus_group(group_df: pd.DataFrame, target: int) -> pd.Series:
        """
        Applies the downsampling logic to a single nucleus group (Pandas DataFrame).
        """
        gene_counts = group_df['feature_name'].value_counts()
        total_transcripts = gene_counts.sum()
        
        if total_transcripts == 0 or len(gene_counts) == 0:
            return pd.Series(dtype=np.int64)
        
        if target == 0:
            return pd.Series(dtype=np.int64)
        
        probabilities = gene_counts / total_transcripts
        
        if probabilities.isna().any():
            logger.warning(f"NaN probabilities detected in group with {total_transcripts} transcripts")
            return pd.Series(dtype=np.int64)
        
        p_normalized = probabilities.values / np.sum(probabilities.values)
        
        if not np.allclose(np.sum(p_normalized), 1.0, atol=1e-7):
            logger.warning(f"Probabilities don't sum to 1: {np.sum(p_normalized)}")
            p_normalized = p_normalized / np.sum(p_normalized)
        
        if np.any(p_normalized < 0) or np.any(p_normalized > 1) or np.any(np.isnan(p_normalized)):
            logger.warning(f"Invalid probabilities detected: min={np.min(p_normalized)}, max={np.max(p_normalized)}")
            return pd.Series(dtype=np.int64)
        
        p_normalized = p_normalized.astype(np.float64)
        new_counts_array = multinomial.rvs(n=target, p=p_normalized, size=1)[0]
        new_counts_series = pd.Series(new_counts_array, index=gene_counts.index)
        new_counts_series = new_counts_series[new_counts_series > 0]
        return new_counts_series
    
    groupby_obj = intersected_gdf.groupby('nucleus_id_new')
    expanded_median = groupby_obj.size().median().astype(int)

    normalized_counts_series = groupby_obj.apply(
        normalize_nucleus_group, target=expanded_median
    ).rename("counts")

    normalized_counts_series = normalized_counts_series[normalized_counts_series > 0]
    logger.info("Normalization within buffered regions complete.")
    normalized_df = normalized_counts_series.reset_index()

    gene_to_nuclei_series = normalized_df.groupby('feature_name')['nucleus_id_new'].apply(set)

    buffered_gene_nuclei_dict = gene_to_nuclei_series.to_dict()

    buffered_conditional_coexpression = pd.DataFrame(index=buffered_gene_nuclei_dict.keys(), columns=buffered_gene_nuclei_dict.keys())
    logger.info("Computing buffered conditional co-expression scores.")

    for i, j in itertools.permutations(buffered_gene_nuclei_dict.keys(), 2):
        intersection = buffered_gene_nuclei_dict[i].intersection(buffered_gene_nuclei_dict[j])
        score = len(intersection) / len(buffered_gene_nuclei_dict[j])
        buffered_conditional_coexpression.loc[i, j] = score
    logger.info("Buffered conditional co-expression scores computed.")
    return buffered_conditional_coexpression

def get_spurious_pairs_scores(nucleus_adata: sc.AnnData, transcripts: pd.DataFrame, nucleus_gdf: gpd.GeoDataFrame, buffer_size: float | pd.Series, return_nucleus_coexpression: bool = False) -> pd.DataFrame:
    nuclear_conditional_coexpression = find_conditional_coexpression(nucleus_adata)
    transcripts = transcripts.loc[transcripts["feature_name"].isin(nucleus_adata.var_names)]

    logger.info("Computing buffered conditional co-expression.")
    buffered_conditional_coexpression = get_expansion_conditional_coexpression(
        transcripts,
        nucleus_gdf,
        buffer_size,
        genes=nucleus_adata.var_names.tolist() # CHANGE TO REMOVE NON GENE TRANSCRIPTS
    )
    logger.info("Calculating spurious gene pair scores.")
    common_genes = nuclear_conditional_coexpression.columns.intersection(buffered_conditional_coexpression.columns)
    nuclear_conditional_coexpression = nuclear_conditional_coexpression.loc[common_genes, common_genes]

    nuclear_conditional_coexpression = nuclear_conditional_coexpression.fillna(0)
    buffered_conditional_coexpression = buffered_conditional_coexpression.fillna(0)
    spurious_score = buffered_conditional_coexpression.div(nuclear_conditional_coexpression).fillna(0.0).replace([np.inf, -np.inf], 0)

    spurious_score_stack = spurious_score.stack()
    spurious_ranked = spurious_score_stack.rename('coexpression').reset_index()
    spurious_ranked.columns = ['Gene1', 'Gene2', 'conditional_coexpression']
    spurious_ranked = spurious_ranked.sort_values(by='conditional_coexpression', ascending=False)
    spurious_ranked["log_conditional_coexpression"] = np.log2(spurious_ranked["conditional_coexpression"])
    
    if return_nucleus_coexpression:
        return spurious_ranked, nuclear_conditional_coexpression
    
    return spurious_ranked

def get_gdf_from_vertices(nucleus_boundaries_df: pd.DataFrame) -> gpd.GeoDataFrame:
    logger.info
    geometry_series = nucleus_boundaries_df.groupby('cell_id').apply(
        lambda group: list(zip(group['vertex_x'], group['vertex_y']))
    )
    polygon_geometries = geometry_series.apply(Polygon)
    polygon_geometries = geometry_series.apply(Polygon)
    nucleus_gdf = gpd.GeoDataFrame(
        polygon_geometries, 
        geometry=polygon_geometries.name, 
    )

    nucleus_gdf =  nucleus_gdf.rename(columns={0: 'geometry'})
    nucleus_gdf = nucleus_gdf.set_geometry('geometry')
    logger.info("Converted nucleus boundaries to GeoDataFrame.")
    return nucleus_gdf

def get_spurious_pairs_scores_from_xenium(xenium_dir_path: os.PathLike) -> pd.DataFrame:
    xenium_dir_path = Path(xenium_dir_path)
    transcripts = pd.read_parquet(xenium_dir_path / "raw/transcripts.parquet")
    nucleus_boundaries_df = pd.read_parquet(xenium_dir_path / "raw/nucleus_boundaries.parquet")
    nucleus_gdf = get_gdf_from_vertices(nucleus_boundaries_df)

    nuclear_transcripts = transcripts[transcripts["overlaps_nucleus"] == 1]
    nuclear_expr_df = nuclear_transcripts.groupby(["cell_id", "feature_name"]).size().reset_index(name="counts").pivot(index="cell_id", columns="feature_name", values="counts").fillna(0).astype(int)

    nucleus_adata = sc.AnnData(
        X=csr_matrix(nuclear_expr_df.to_numpy()),
        obs=pd.DataFrame(index=nuclear_expr_df.index),
        var=pd.DataFrame(index=nuclear_expr_df.columns)
    )

    buffer_size = 2.0 # EXAMPLE: NOT RIGOROUSLY CHOSEN

    spurious_pairs_scores, nuclear_conditional_coexpression = get_spurious_pairs_scores(
        nucleus_adata,
        transcripts,
        nucleus_gdf,
        buffer_size,
        return_nucleus_coexpression=True
    )

    return spurious_pairs_scores, nuclear_conditional_coexpression, nucleus_adata

def get_segmentation_relative_spurious_scores(xenium_dir_path: os.PathLike, segmentation_adata: sc.AnnData, cutoff: float) -> pd.DataFrame:
    xenium_dir_path = "/data1/peerd/moormana/data/spatial_benchmarking/ist/datasets/xenium_human_colon_healthy_v1"
    segmentation_adata = sc.read_h5ad("/data1/peerd/moormana/data/spatial_benchmarking/ist/datasets/xenium_human_colon_healthy_v1/analysis/h5ad/proseg_cell_id_processed.h5ad")
    cutoff = 4.0 # EXAMPLE: NOT RIGOROUSLY CHOSEN
    xenium_dir_path = Path(xenium_dir_path)
    spurious_pairs_scores, nuclear_conditional_coexpression, nucleus_adata = get_spurious_pairs_scores_from_xenium(xenium_dir_path)
    nuclear_conditional_coexpression = find_conditional_coexpression(
        nucleus_adata
    )

    spurious_pairs_scores = spurious_pairs_scores.copy()
    spurious_pairs = spurious_pairs_scores["log_conditional_coexpression"] > cutoff
    segmentation_conditional_coexpression = find_conditional_coexpression(segmentation_adata)

    nuclear_conditional_coexpression = nuclear_conditional_coexpression.fillna(0)
    segmentation_condition_normalized = segmentation_conditional_coexpression.copy()
    segmentation_condition_normalized = segmentation_condition_normalized.fillna(0)
    segmentation_condition_normalized = segmentation_condition_normalized.div(nuclear_conditional_coexpression).fillna(0).replace([np.inf, -np.inf], 0)

    segmentation_stacked = segmentation_condition_normalized.stack()
    segmentation_ranked = segmentation_stacked.rename('coexpression').reset_index()
    segmentation_ranked.columns = ['Gene1', 'Gene2', 'conditional_coexpression']
    segmentation_ranked = segmentation_ranked.sort_values(by='conditional_coexpression', ascending=False)
    segmentation_ranked["log_conditional_coexpression"] = np.log2(segmentation_ranked["conditional_coexpression"])
    segmentation_ranked = segmentation_ranked.loc[np.isfinite(segmentation_ranked["log_conditional_coexpression"])]
    segmentation_ranked["combined_gene"] = segmentation_ranked["Gene1"] + "_" + segmentation_ranked["Gene2"]
    segmentation_ranked = segmentation_ranked.loc[segmentation_ranked["combined_gene"].isin(spurious_pairs["gene_combined"])]
    segmentation_ranked = segmentation_ranked.drop(columns=["combined_gene"])
    return segmentation_ranked, spurious_pairs_scores