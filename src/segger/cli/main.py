from cyclopts import App
from segger import configure_memory
from .segment import segment
from .debug import debug
from .export import export

configure_memory()

app = App(name="Segger")

app.command(segment)
app.command(debug)
app.command(export)
