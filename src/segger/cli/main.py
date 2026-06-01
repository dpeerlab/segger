from cyclopts import App
from .segment import segment
from .validate import validate
from .debug import debug

# CLI App
app = App(name="Segger")

# Main segmentation
app.command(segment)

# Segmentation-quality metrics
app.command(validate)

# Debugging utilities
app.command(debug)
