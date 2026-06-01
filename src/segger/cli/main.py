from cyclopts import App
from .segment import segment
from .predict import predict
from .debug import debug

# CLI App
app = App(name="Segger")

# Main segmentation
app.command(segment)

# Prediction-only from a trained checkpoint
app.command(predict)

# Debugging utilities
app.command(debug)
