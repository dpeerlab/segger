"""Sphinx configuration for the segger docs."""

project = "segger"
author = "Elyas Heidari, Andrew Moorman"
maintainer = "Tobias Krause"

extensions = []

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_show_copyright = False
