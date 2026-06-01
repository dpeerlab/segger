"""Run segger prediction only (no training) from a saved checkpoint."""

import os
import logging
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter, Group, validators

from ..utils import setup_logging
setup_logging(level=os.environ.get("LOG_LEVEL", "WARNING"))


group_io = Group(name="I/O", help="Inputs/outputs.", sort_key=0)


app_predict = App(
    name="predict",
    help="Run prediction-only segmentation from a trained checkpoint.",
)


@app_predict.command(name="predict")
def predict(
    checkpoint_path: Annotated[Path, Parameter(
        alias="-c",
        help="Path to the Lightning checkpoint (.ckpt) saved during training.",
        validator=validators.Path(exists=True, dir_okay=False),
        group=group_io,
    )],

    output_directory: Annotated[Path, Parameter(
        alias="-o",
        help="Path to write predictions / segmentation outputs.",
        validator=validators.Path(exists=True, dir_okay=True),
        group=group_io,
    )],

    input_directory: Annotated[Path | None, Parameter(
        alias="-i",
        help="Override input dataset directory (defaults to the one stored in the checkpoint).",
        validator=validators.Path(exists=True, dir_okay=True),
        group=group_io,
    )] = None,

    save_anndata: Annotated[bool, Parameter(
        help="Save the AnnData output alongside the segmentation parquet.",
        group=group_io,
    )] = True,
):
    """Predict segmentation from a saved checkpoint, skipping training."""
    logger = logging.getLogger(__name__)

    from lightning.pytorch.plugins.environments import SLURMEnvironment
    SLURMEnvironment.detect = lambda: False

    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch import Trainer

    from ..data import ISTDataModule, ISTSegmentationWriter
    from ..models import LitISTEncoder

    os.makedirs(output_directory, exist_ok=True)

    logger.debug("Loading model and data module from checkpoint: %s", checkpoint_path)
    model = LitISTEncoder.load_from_checkpoint(checkpoint_path)
    datamodule_kwargs = {}
    if input_directory is not None:
        datamodule_kwargs["input_directory"] = input_directory
    datamodule = ISTDataModule.load_from_checkpoint(
        checkpoint_path,
        **datamodule_kwargs,
    )

    writer = ISTSegmentationWriter(
        output_directory,
        save_anndata=save_anndata,
    )
    trainer = Trainer(
        logger=CSVLogger(output_directory),
        reload_dataloaders_every_n_epochs=1,
        callbacks=[writer],
    )

    logger.debug("Running prediction...")
    trainer.predict(model=model, datamodule=datamodule)
