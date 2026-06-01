from cyclopts import App
from .segment import segment
from .export import export
from .debug import debug

# CLI App
app = App(name="Segger")

# Main segmentation
app.command(segment)

# Export a segmentation to Xenium Explorer / scverse formats
app.command(export)

# Debugging utilities
app.command(debug)
