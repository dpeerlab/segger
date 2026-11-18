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

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/dpeerlab/segger",
    "show_toc_level": 2,
    "navigation_depth": 3,
    "secondary_sidebar_items": [],
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
# the sections of the current page, in the left sidebar (the top bar only links to top-level pages)
html_sidebars = {"**": ["sidebar-nav-bs", "page-toc"]}
html_show_copyright = False
