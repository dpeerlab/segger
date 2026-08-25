"""Sphinx configuration for the segger docs."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "segger"
copyright = "Dana Pe'er Lab"
author = "Elyas Heidari, Andrew Moorman"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Autodoc needs to import segger's modules, which unconditionally import the CUDA/RAPIDS
# stack (torch, cupy, rmm, ...) at module load time. Mock these so the docs build without
# a GPU environment; real signatures/defaults/docstrings are still introspected correctly.
autodoc_mock_imports = [
    "anndata", "cudf", "cugraph", "cuml", "cupy", "cupyx", "cuspatial", "cv2",
    "cyclopts", "docstring_parser", "geopandas", "lightning", "numba", "numpy",
    "pandas", "polars", "pyarrow", "rmm", "scanpy", "scipy", "shapely", "skimage",
    "sklearn", "tifffile", "torch", "torch_geometric", "torch_scatter", "torchvision",
    "tqdm", "typer",
]
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
