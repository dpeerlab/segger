"""End-to-end tests for the ``segger export`` CLI command."""

from pathlib import Path

import polars as pl
import pytest


def _write_source(dir_: Path, n: int = 6) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "transcript_id": [f"t{i}" for i in range(n)],
            "x_location": [float(i) for i in range(n)],
            "y_location": [0.0] * n,
            "z_location": [0.0] * n,
            "feature_name": (["g1", "g2"] * n)[:n],
        }
    ).write_parquet(dir_ / "transcripts.parquet")
    return dir_


def _write_segmentation(path: Path, cell_id_column: str = "segger_cell_id") -> Path:
    pl.DataFrame(
        {
            "row_index": [0, 1, 2, 3, 4, 5],
            cell_id_column: [1, 1, 2, 2, None, None],
            "segger_similarity": [0.9, 0.8, 0.7, 0.6, 0.0, 0.0],
            "similarity_threshold": [0.5] * 6,
        },
        schema_overrides={cell_id_column: pl.Int64},
    ).write_parquet(path)
    return path


def test_export_cli_xenium_transcript_assignment(tmp_path):
    pytest.importorskip("geopandas")
    pytest.importorskip("shapely")
    pytest.importorskip("rtree")
    from segger.cli.export import export

    src = _write_source(tmp_path / "src")
    seg = _write_segmentation(tmp_path / "seg.parquet")
    out = tmp_path / "out"

    export(
        segmentation_path=seg,
        source_path=src,
        output_directory=out,
        format="xenium",
        xenium_mode="transcript_assignment",
        num_workers=1,
    )

    csv = pl.read_csv(out / "segmentation.csv")
    assert {"transcript_id", "cell", "is_noise"} <= set(csv.columns)
    assert csv.height == 6  # one row per source transcript
    assert (out / "segmentation_polygons.json").exists()


def test_export_cli_resolves_cell_id_alias(tmp_path):
    pytest.importorskip("geopandas")
    pytest.importorskip("shapely")
    pytest.importorskip("rtree")
    from segger.cli.export import export

    src = _write_source(tmp_path / "src")
    # Segmentation stores the assignment under a non-standard column name.
    seg = _write_segmentation(tmp_path / "seg.parquet", cell_id_column="seg_cell_id")
    out = tmp_path / "out"

    export(
        segmentation_path=seg,
        source_path=src,
        output_directory=out,
        format="xenium",
        xenium_mode="transcript_assignment",
    )
    assert (out / "segmentation.csv").exists()


def test_export_cli_rejects_unknown_suffix(tmp_path):
    from segger.cli.export import export

    src = _write_source(tmp_path / "src")
    bad = tmp_path / "seg.txt"
    bad.write_text("not a segmentation")
    with pytest.raises(ValueError):
        export(segmentation_path=bad, source_path=src, output_directory=tmp_path / "out")
