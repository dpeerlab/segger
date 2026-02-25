"""Run only prediction, followed by segmentation."""

import os

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

    # load objects (analogous to segment.py)
    csvlogger = CSVLogger(path_outputs)
    writer = ISTSegmentationWriter(path_outputs, debug=True)

    trainer = Trainer(logger=csvlogger, reload_dataloaders_every_n_epochs=1, callbacks=[writer],)
    datamodule = ISTDataModule.load_from_checkpoint(path_checkpoint)
    model = LitISTEncoder.load_from_checkpoint(path_checkpoint)

    # predict (and save results via callback)
    predictions = trainer.predict(model=model, datamodule=datamodule)