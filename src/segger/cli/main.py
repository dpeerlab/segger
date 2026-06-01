from cyclopts import App
from .segment import segment, merge_splits
from .debug import debug

# CLI App
app = App(name="Segger")

# Main segmentation
app.command(segment)

# Merge per-subset gene-split outputs (final step of a VRAM-bounded split run)
app.command(merge_splits)

# Debugging utilities
app.command(debug)
