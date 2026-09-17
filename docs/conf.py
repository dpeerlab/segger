"""Sphinx configuration for the segger docs."""

project = "segger"
author = "Elyas Heidari, Andrew Moorman"
maintainer = "Tobias Krause"

extensions = ["nbsphinx"]

# Adds a "View on GitHub / Download notebook" link at the top of every notebook page.
nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}
{% set filename = docname.split('/')[-1] %}

.. raw:: html

    <p>
      <a href="https://github.com/dpeerlab/segger/blob/main/{{ docname }}">View on GitHub</a>
      &middot;
      <a href="{{ filename }}">Download notebook</a>
    </p>
"""

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_show_copyright = False
