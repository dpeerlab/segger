"""Smoke tests for the 10x ``import-segmentation`` export path."""

import json
from pathlib import Path

import polars as pl
import pytest

from segger.export.xenium_import import (
    _build_assignment,
    _load_source_transcripts,
    build_import_command,
    export_xenium_import,
    write_baysor_csv,
)


def _write_source(tmp_path: Path, df: pl.DataFrame) -> Path:
    df.write_parquet(tmp_path / "transcripts.parquet")
    return tmp_path


# --- Lightweight CSV path (polars only) --------------------------------------


def test_baysor_csv_assignment_and_noise(tmp_path):
    # 6 raw transcripts; segmentation only covers row_index 0..4 (row 5 is filtered out).
    src = _write_source(
        tmp_path / "src",
        pl.DataFrame(
            {
                "transcript_id": [f"t{i}" for i in range(6)],
                "x_location": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "y_location": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "z_location": [0.0] * 6,
                "feature_name": ["g1", "g1", "g2", "g2", "g3", "g3"],
            }
        ),
    )
    seg = pl.DataFrame(
        {
            "row_index": [0, 1, 2, 3, 4],
            "segger_cell_id": [1, 1, 2, None, 2],
            "segger_similarity": [0.9] * 5,
        },
        schema_overrides={"segger_cell_id": pl.Int64},
    )

    raw = _load_source_transcripts(
        src,
        transcript_id_column="transcript_id",
        x_column="x_location",
        y_column="y_location",
        z_column="z_location",
        feature_column="feature_name",
    )
    assignment = _build_assignment(seg, raw, cell_id_column="segger_cell_id")
    csv_path = write_baysor_csv(
        assignment,
        tmp_path / "out",
        transcript_id_column="transcript_id",
        x_column="x_location",
        y_column="y_location",
        z_column="z_location",
        feature_column="feature_name",
    )

    df = pl.read_csv(csv_path)
    # Required Baysor columns
    assert {"transcript_id", "cell", "is_noise"} <= set(df.columns)
    assert df.height == 6  # all raw transcripts represented
    # row 3 (null assignment) and row 5 (not in segmentation) are noise
    noise = df.filter(pl.col("is_noise"))
    assert set(noise["transcript_id"].to_list()) == {"t3", "t5"}
    # assigned cells are exactly {1, 2}
    assigned = df.filter(~pl.col("is_noise"))
    assert set(assigned["cell"].to_list()) == {1, 2}


def test_min_similarity_marks_low_confidence_as_noise(tmp_path):
    src = _write_source(
        tmp_path / "src",
        pl.DataFrame(
            {
                "transcript_id": ["a", "b"],
                "x_location": [0.0, 1.0],
                "y_location": [0.0, 0.0],
                "z_location": [0.0, 0.0],
                "feature_name": ["g", "g"],
            }
        ),
    )
    # 'keep' precomputed by the CLI: second transcript fails the threshold.
    seg = pl.DataFrame(
        {
            "row_index": [0, 1],
            "segger_cell_id": [1, 1],
            "segger_similarity": [0.9, 0.1],
            "keep": [True, False],
        },
        schema_overrides={"segger_cell_id": pl.Int64},
    )
    raw = _load_source_transcripts(
        src,
        transcript_id_column="transcript_id",
        x_column="x_location",
        y_column="y_location",
        z_column="z_location",
        feature_column="feature_name",
    )
    assignment = _build_assignment(seg, raw, cell_id_column="segger_cell_id")
    rows = {r["transcript_id"]: r["is_noise"] for r in assignment.to_dicts()}
    assert rows["a"] is False
    assert rows["b"] is True


def test_build_import_command_forms():
    ta = build_import_command(
        mode="transcript_assignment",
        run_id="demo",
        source_path=Path("/data/xenium"),
        files={"csv": Path("/o/segmentation.csv"), "viz": Path("/o/segmentation_polygons.json")},
        units="microns",
    )
    assert "xeniumranger import-segmentation" in ta
    assert "--transcript-assignment=segmentation.csv" in ta
    assert "--viz-polygons=segmentation_polygons.json" in ta
    assert "--units=microns" in ta

    geo = build_import_command(
        mode="geojson",
        run_id="demo",
        source_path=Path("/data/xenium"),
        files={"cells": Path("/o/polygon.geojson")},
        units="microns",
    )
    assert "--cells=polygon.geojson" in geo


# --- Full path incl. polygon generation (needs the geo stack) ----------------


def test_export_both_writes_valid_polygons(tmp_path):
    pytest.importorskip("geopandas")
    pytest.importorskip("shapely")
    pytest.importorskip("rtree")

    # Three cells, each a 4-point square cluster (>=3 non-collinear points).
    def square(cx, cy):
        return [(cx, cy), (cx + 1, cy), (cx + 1, cy + 1), (cx, cy + 1)]

    pts = square(0, 0) + square(20, 0) + square(0, 20)
    cells = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    src = _write_source(
        tmp_path / "src",
        pl.DataFrame(
            {
                "transcript_id": [f"t{i}" for i in range(12)],
                "x_location": [float(x) for x, _ in pts],
                "y_location": [float(y) for _, y in pts],
                "z_location": [0.0] * 12,
                "feature_name": ["g"] * 12,
            }
        ),
    )
    seg = pl.DataFrame(
        {"row_index": list(range(12)), "segger_cell_id": cells, "segger_similarity": [0.9] * 12},
        schema_overrides={"segger_cell_id": pl.Int64},
    )

    out = tmp_path / "out"
    written = export_xenium_import(
        seg, src, out, mode="both", n_jobs=1, run_id="demo", progress=False
    )

    assert written["segmentation_csv"].exists()
    assert written["viz_polygons"].exists()
    assert written["cell_geojson"].exists()

    csv = pl.read_csv(written["segmentation_csv"])
    assigned_cells = set(csv.filter(~pl.col("is_noise"))["cell"].to_list())

    fc = json.loads(written["viz_polygons"].read_text())
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) >= 1
    for feat in fc["features"]:
        # Every visualized polygon must correspond to a cell with transcripts.
        assert feat["properties"]["cell"] in assigned_cells
        ring = feat["geometry"]["coordinates"][0]
        assert len(ring) >= 4  # >=3 vertices + closure (Explorer crashes on fewer)
