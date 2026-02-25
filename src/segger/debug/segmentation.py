"""
Run only segmentation.
"""

import pickle
import anndata as ad

def run_segmentation_only(
    path_adata,
    path_predictions,
    path_outputs,
):
    """Run segmentation only."""

    # imports
    from ..data.writer import ISTSegmentationWriter
    from lightning.pytorch.plugins.environments import SLURMEnvironment
    SLURMEnvironment.detect = lambda: False

    # Load data
    writer = ISTSegmentationWriter(path_outputs, debug=True)
    adata = ad.read_h5ad(path_adata)
    with open(path_predictions, "rb") as f:
        predictions = pickle.load(f)

    # Predict and write output
    segmentation = writer.assign_transcripts_to_cells(
        obs=adata.obs,
        predictions=predictions,
    )

    segmentation.write_parquet(path_outputs / 'segger_segmentation.parquet')

