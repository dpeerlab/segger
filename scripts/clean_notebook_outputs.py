#!/usr/bin/env python
"""Strip notebook cell outputs down to stdout text and images; drop stderr/errors/execution counts."""

import sys

import nbformat

KEEP_STREAM_NAMES = {"stdout"}
KEEP_DATA_KEYS = {"text/plain", "text/html", "image/png", "image/jpeg", "image/svg+xml"}


def _keep_output(output: dict) -> bool:
    if output["output_type"] == "stream":
        return output["name"] in KEEP_STREAM_NAMES
    if output["output_type"] in ("execute_result", "display_data"):
        data = {k: v for k, v in output.get("data", {}).items() if k in KEEP_DATA_KEYS}
        if not data:
            return False
        output["data"] = data
        output["metadata"] = {}
        return True
    return False  # drop "error" outputs (tracebacks)


def clean_notebook(path: str) -> None:
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell["cell_type"] != "code":
            continue
        cell["execution_count"] = None
        cell["outputs"] = [o for o in cell.get("outputs", []) if _keep_output(o)]
        cell["metadata"].pop("collapsed", None)
        cell["metadata"].pop("scrolled", None)
    nb["metadata"].pop("widgets", None)
    nbformat.write(nb, path)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        clean_notebook(path)
