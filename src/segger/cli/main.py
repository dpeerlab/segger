from cyclopts import App
from .segment import segment
from .debug import debug
from .export import export

app = App(name="Segger")

app.command(segment)
app.command(debug)
app.command(export)
