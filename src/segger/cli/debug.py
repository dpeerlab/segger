"""
Debugging utilities for Segger.
"""

import os
from pathlib import Path
from typing_extensions import Annotated
from cyclopts import App, Parameter

from ..debug.segmentation import run_segmentation_only
from ..debug.prediction import run_prediction_only
from ..utils import setup_logging

debug = App(name="debug", help="Utilities for debugging and testing individual components.")

@debug.command(name="segment-only")
def segment_only_cli(
    path_adata: Annotated[Path, Parameter(
        help="Path to input AnnData object.",
    )],
    path_predictions: Annotated[Path, Parameter(
        help="Path to write predictions.",
    )],
    path_outputs: Annotated[Path, Parameter(
        help="Path to write outputs.",
    )],
):
    """Run prediction only."""
    run_segmentation_only(
        path_adata=path_adata,
        path_predictions=path_predictions,
        path_outputs=path_outputs,
    )

@debug.command(name="predict-only")
def predict_only_cli(
    path_checkpoint: Annotated[Path, Parameter(
        help="Path to trainer checkpoint object.",
    )],
    path_outputs: Annotated[Path, Parameter(
        help="Path to write outputs.",
    )],
):
    """Run prediction only."""
    setup_logging(level="DEBUG", debug=True)
    run_prediction_only(
        path_checkpoint=path_checkpoint,
        path_outputs=path_outputs,
    )