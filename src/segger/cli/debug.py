"""
Debugging utilities for Segger.
"""

from pathlib import Path
from typing_extensions import Annotated
from cyclopts import App, Parameter
from ..debug.prediction import run_prediction_only

debug = App(name="debug", help="Utilities for debugging and testing individual components.")

@debug.command(name="predict-only")
def predict_only(
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
    run_prediction_only(
        path_adata=path_adata,
        path_predictions=path_predictions,
        path_outputs=path_outputs,
    )