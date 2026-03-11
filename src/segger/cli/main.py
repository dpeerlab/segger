from cyclopts import App
from .segment import segment
from .debug import debug
from .debug_rapids import debug_rapids

# CLI App
app = App(name="Segger")

# Main segmentation
app.command(segment)

# Debugging utilities
app.command(debug)

app.command(debug_rapids)