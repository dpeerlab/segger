"""Run only prediction, followed by segmentation."""

import os
import logging
import pickle
from pathlib import Path
from types import SimpleNamespace


def _patch_load_from_cache():
    """Monkey-patch ISTDataModule.load to restore from `debug_dir` cache.

    The original `load()` always re-runs setup_anndata + setup_heterodata + tiling
    (~2 h on cervical). For predict-only that's wasted: when --debug was set on
    the original run, those outputs are already on disk in `debug_dir` as
    `data.pt`, `tiles.pkl`, and `adata_debug.h5ad`. Restore from them instead.
    """
    import scanpy as sc
    import polars as pl
    import torch
    from segger.data import ISTDataModule
    from segger.io.fields import StandardTranscriptFields
    from segger.io import get_preprocessor

    logger = logging.getLogger(__name__)
    original_load = ISTDataModule.load

    def cached_load(self):
        d = Path(self.debug_dir) if self.debug_dir is not None else None
        cached = {
            "data":  d / "data.pt"           if d else None,
            "tiles": d / "tiles.pkl"         if d else None,
            "adata": d / "adata_debug.h5ad"  if d else None,
        }
        if d is None or not all(p.exists() for p in cached.values()):
            logger.info("Cached artifacts not found; falling back to full rebuild.")
            return original_load(self)

        logger.info(f"Restoring cached datamodule state from {d}")
        tx_fields = StandardTranscriptFields()

        # Raw transcripts/boundaries — only used by writer.write_anndata; cheap to re-read.
        pp = get_preprocessor(self.input_directory)
        self.tx = pp.transcripts
        self.bd = pp.boundaries

        # Cached artifacts
        self.ad   = sc.read_h5ad(cached["adata"])
        self.data = torch.load(cached["data"], weights_only=False)
        with open(cached["tiles"], "rb") as f:
            tiles = pickle.load(f)
        # Predict only accesses `self.tiling.tiles[idx]`; a SimpleNamespace shell suffices.
        self.tiling = SimpleNamespace(tiles=tiles)

        # Model-side embeddings/similarities — rebuilt from adata
        self.tx_embedding = (
            pl.from_numpy(self.ad.varm['X_corr'])
            .cast(pl.Float32)
            .with_columns(pl.Series(self.ad.var.index).alias(tx_fields.feature))
        )
        self.tx_similarity = torch.tensor(self.ad.uns['gene_cluster_similarities'])
        self.bd_similarity = torch.tensor(self.ad.uns['cell_cluster_similarities'])
        logger.debug("Data loading is complete (cache).")

    ISTDataModule.load = cached_load


def run_prediction_only(
    path_checkpoint,
    path_outputs,
):
    from segger.data import ISTDataModule
    from segger.data import ISTSegmentationWriter
    from segger.models import LitISTEncoder

    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch import Trainer

    from lightning.pytorch.plugins.environments import SLURMEnvironment
    SLURMEnvironment.detect = lambda: False

    os.makedirs(path_outputs, exist_ok=True)

    # Skip the full setup_anndata/setup_heterodata/tiling rebuild
    _patch_load_from_cache()

    # load objects (analogous to segment.py)
    csvlogger = CSVLogger(path_outputs)
    writer = ISTSegmentationWriter(path_outputs, debug=True)

    trainer = Trainer(logger=csvlogger, reload_dataloaders_every_n_epochs=1, callbacks=[writer], devices=1,)
    datamodule = ISTDataModule.load_from_checkpoint(path_checkpoint)
    model = LitISTEncoder.load_from_checkpoint(path_checkpoint)

    # predict (and save results via callback)
    predictions = trainer.predict(model=model, datamodule=datamodule)