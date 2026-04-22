from cyclopts import App
from .segment import segment
from .debug import debug

# CLI App
app = App(name="Segger")

# Main segmentation
app.command(segment)

# Debugging utilities
app.command(debug)
