"""Field definitions for spatial transcriptomics platforms.

This module defines dataclasses that map platform-specific column names
to standardized internal field names. Each platform (Xenium, CosMx, MERSCOPE)
has its own field definitions, which are normalized to StandardTranscriptFields
and StandardBoundaryFields during preprocessing.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class XeniumTranscriptFields:
    """Field mappings for 10x Genomics Xenium transcript data.

    Xenium provides transcript-level quality scores (QV) in Phred scale,
    and z-coordinates for 3D spatial analysis.
    """
    filename: str = 'transcripts.parquet'
    x: str = 'x_location'
    y: str = 'y_location'
    z: str = 'z_location'  # Z-coordinate (microns)
    feature: str = 'feature_name'
    cell_id: str = 'cell_id'
    null_cell_id: str = 'UNASSIGNED'
    compartment: str = 'overlaps_nucleus'
    nucleus_value: int = 1
    quality: str = 'qv'  # Phred-scaled quality value (0-40)
    filter_substrings: List[str] = field(default_factory=lambda: [
        'NegControlProbe_*',
        'antisense_*',
        'NegControlCodeword*',
        'BLANK_*',
        'DeprecatedCodeword_*',
        'UnassignedCodeword_*',
    ])
    is_gene: str = 'is_gene'  # filter field in newer Xenium outputs

@dataclass
class XeniumBoundaryFields:
    """Field mappings for 10x Genomics Xenium boundary data."""
    cell_filename: str = 'cell_boundaries.parquet'
    nucleus_filename: str = 'nucleus_boundaries.parquet'
    x: str = 'vertex_x'
    y: str = 'vertex_y'
    id: str = 'cell_id'


@dataclass
class MerscopeTranscriptFields:
    """Field mappings for Vizgen MERSCOPE transcript data.

    MERSCOPE uses blank barcodes (BLANK_*) for quality control instead of
    per-transcript QV scores. Z-coordinates are available for 3D analysis.
    """
    filename: str = 'detected_transcripts.csv'
    x: str = 'global_x'
    y: str = 'global_y'
    z: str = 'global_z'  # Z-coordinate (microns)
    feature: str = 'gene'
    cell_id: str = 'cell_id'
    # MERSCOPE has no transcript-level QV; uses blank codes for FDR
    quality: Optional[str] = None
    filter_substrings: List[str] = field(default_factory=lambda: [
        'BLANK_*',  # Blank barcodes for FDR calculation
    ])

@dataclass
class MerscopeBoundaryFields:
    """Field mappings for Vizgen MERSCOPE boundary data."""
    cell_filename: str = 'cell_boundaries.parquet'
    nucleus_filename: str = 'nucleus_boundaries.parquet'
    id: str = 'EntityID'


@dataclass
class CosMxTranscriptFields:
    """Field mappings for NanoString CosMx transcript data.

    CosMx has no per-transcript quality score. Quality control is done via:
    - Control probe removal (Negative*, SystemControl*, NegPrb*)
    - Cell-level thresholds (tx count >= 30, size < 5x mean)

    Z-coordinates are available for 3D spatial analysis.
    """
    filename: str = '*_tx_file.csv'
    x: str = 'x_global_px'
    y: str = 'y_global_px'
    z: str = 'z'  # Z-coordinate (z-stack index or microns)
    feature: str = 'target'
    cell_id: str = 'cell'
    compartment: str = 'CellComp'
    nucleus_value: str = 'Nuclear'
    membrane_value: str = 'Membrane'
    cytoplasmic_value: str = 'Cytoplasm'
    extracellular_value: str = 'None'
    # CosMx has no transcript-level QV; uses control probes
    quality: Optional[str] = None
    filter_substrings: List[str] = field(default_factory=lambda: [
        'Negative*',
        'SystemControl*',
        'NegPrb*',
    ])

@dataclass
class CosMxBoundaryFields:
    """Field mappings for NanoString CosMx boundary data."""
    id: str = 'cell_id'
    cell_labels_dirname: str = 'CellLabels'
    compartment_labels_dirname: str = 'CompartmentLabels'
    fov_positions_filename: str = '*fov_positions_file.csv'
    extracellular_value: int = 0
    nucleus_value: int = 1
    membrane_value: int = 2
    cytoplasmic_value: int = 3
    mpp: float = 0.12028


@dataclass
class StandardTranscriptFields:
    """Standardized field names for internal transcript representation.

    All platform-specific transcript data is normalized to these field names
    during preprocessing. The z-coordinate is optional (None for 2D data).
    """
    filename: str = 'transcripts.parquet'
    row_index: str = 'row_index'
    x: str = 'x'
    y: str = 'y'
    z: str = 'z'  # Z-coordinate (optional, None for 2D data)
    feature: str = 'feature_name'
    cell_id: str = 'cell_id'
    quality: str = 'qv'  # Quality score (optional, platform-specific)
    compartment: str = 'cell_compartment'
    extracellular_value: int = 0
    cytoplasmic_value: int = 1
    nucleus_value: int = 2

@dataclass
class StandardBoundaryFields:
    """Standardized field names for internal boundary representation.

    All platform-specific boundary data is normalized to these field names
    during preprocessing. The z_slice field is optional for 3D data.
    """
    filename: str = 'boundaries.parquet'
    id: str = 'cell_id'
    boundary_type: str = 'boundary_type'
    cell_value: str = 'cell'
    nucleus_value: str = 'nucleus'
    contains_nucleus: str = 'contains_nucleus'
    z_slice: str = 'z_slice'  # Z-coordinate for 3D slices (optional)


@dataclass
class TrainingTranscriptFields(StandardTranscriptFields):
    """Extended transcript fields for training data with encodings and clusters."""
    cell_encoding: str = 'cell_encoding'
    gene_encoding: str = 'gene_encoding'
    cell_cluster: str = 'cell_cluster'
    gene_cluster: str = 'gene_cluster'


@dataclass
class TrainingBoundaryFields(StandardBoundaryFields):
    """Extended boundary fields for training data with encodings and clusters."""
    index: str = 'entity_index'
    cell_encoding: str = 'cell_encoding'
    cell_cluster: str = 'cell_cluster'
