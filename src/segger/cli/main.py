from cyclopts import App
from .segment import segment, merge_splits
from .predict import predict
from .validate import validate
from .export import export
from .debug import debug

# CLI App
app = App(name="Segger")

# Main segmentation
app.command(segment)

# Prediction-only from a trained checkpoint
app.command(predict)

# Segmentation-quality metrics
app.command(validate)

# Merge per-subset gene-split outputs (final step of a VRAM-bounded split run)
app.command(merge_splits)

# Export a segmentation to Xenium Explorer / scverse formats
app.command(export)

# Debugging utilities
app.command(debug)
