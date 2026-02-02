"""Tests for field definitions and defaults."""

from segger.io.fields import (
    XeniumTranscriptFields,
    XeniumBoundaryFields,
    MerscopeTranscriptFields,
    MerscopeBoundaryFields,
    CosMxTranscriptFields,
    CosMxBoundaryFields,
    StandardTranscriptFields,
    StandardBoundaryFields,
    TrainingTranscriptFields,
    TrainingBoundaryFields,
)


def test_standard_transcript_fields_defaults():
    fields = StandardTranscriptFields()
    assert fields.x == "x"
    assert fields.y == "y"
    assert fields.z == "z"
    assert fields.feature == "feature_name"
    assert fields.cell_id == "cell_id"
    assert fields.quality == "qv"
    assert fields.compartment == "cell_compartment"


def test_standard_boundary_fields_defaults():
    fields = StandardBoundaryFields()
    assert fields.id == "cell_id"
    assert fields.boundary_type == "boundary_type"
    assert fields.cell_value == "cell"
    assert fields.nucleus_value == "nucleus"


def test_training_fields_extend_standard_fields():
    tx_fields = TrainingTranscriptFields()
    bd_fields = TrainingBoundaryFields()

    # Base fields still present
    assert tx_fields.x == "x"
    assert tx_fields.y == "y"
    assert bd_fields.id == "cell_id"

    # Training-only fields are present
    assert tx_fields.cell_encoding == "cell_encoding"
    assert tx_fields.gene_encoding == "gene_encoding"
    assert bd_fields.index == "entity_index"
    assert bd_fields.cell_encoding == "cell_encoding"


def test_platform_field_defaults():
    xenium_tx = XeniumTranscriptFields()
    xenium_bd = XeniumBoundaryFields()
    merscope_tx = MerscopeTranscriptFields()
    merscope_bd = MerscopeBoundaryFields()
    cosmx_tx = CosMxTranscriptFields()
    cosmx_bd = CosMxBoundaryFields()

    assert xenium_tx.filename == "transcripts.parquet"
    assert xenium_bd.cell_filename.endswith("cell_boundaries.parquet")
    assert merscope_tx.filename == "detected_transcripts.csv"
    assert merscope_bd.id == "EntityID"
    assert cosmx_tx.filename == "*_tx_file.csv"
    assert cosmx_bd.cell_labels_dirname == "CellLabels"
