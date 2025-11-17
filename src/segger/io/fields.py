# TODO: Add file header description

from dataclasses import dataclass

# TODO: Add description
@dataclass
class XeniumTranscriptFields:
    filename: str = 'transcripts.parquet'
    x: str = 'x_location'
    y: str = 'y_location'
    feature: str = 'feature_name'
    cell_id: str = 'cell_id'
    null_cell_id: str = 'UNASSIGNED'
    compartment: str = 'overlaps_nucleus'
    nucleus_value: int = 1
    quality: str = 'qv'
    filter_substrings = [
        'NegControlProbe_*',
        'antisense_*',
        'NegControlCodeword*',
        'BLANK_*',
        'DeprecatedCodeword_*',
        'UnassignedCodeword_*',
    ]

@dataclass
class XeniumBoundaryFields:
    cell_filename: str = 'cell_boundaries.parquet'
    nucleus_filename: str = 'nucleus_boundaries.parquet'
    x: str = 'vertex_x'
    y: str = 'vertex_y'
    id: str = 'cell_id'


# TODO: Add description
@dataclass
class MerscopeTranscriptFields:
    filename: str = 'detected_transcripts.csv'
    x: str = 'global_x'
    y: str = 'global_y'
    feature: str = 'gene'
    cell_id: str = 'cell_id'

@dataclass
class MerscopeBoundaryFields:
    cell_filename: str = 'cell_boundaries.parquet'
    nucleus_filename: str = 'nucleus_boundaries.parquet'
    id = 'EntityID'


# TODO: Add description
@dataclass
class CosMxTranscriptFields:
    filename: str = '*_tx_file.csv'
    x: str = 'x_global_px'
    y: str = 'y_global_px'
    feature: str = 'target'
    cell_id: str = 'cell'
    compartment: str = 'CellComp'
    nucleus_value: str = 'Nuclear'
    membrane_value: str = 'Membrane'
    cytoplasmic_value: str = 'Cytoplasm'
    extracellular_value: str = 'None'
    filter_substrings = [
        'Negative*',
        'SystemControl*',
        'NegPrb*',
    ]

@dataclass
class CosMxBoundaryFields:
    id: str = 'cell_id'
    cell_labels_dirname: str = 'CellLabels'
    compartment_labels_dirname: str = 'CompartmentLabels'
    fov_positions_filename: str = '*fov_positions_file.csv'
    extracellular_value: int = 0
    nucleus_value: int = 1
    membrane_value: int = 2
    cytoplasmic_value: int = 3
    mpp: float = 0.12028


# TODO: Add description
@dataclass
class StandardTranscriptFields:
    filename: str = 'transcripts.parquet'
    row_index: str = 'row_index'
    x: str = 'x'
    y: str = 'y'
    feature: str = 'feature_name'
    cell_id: str = 'cell_id'
    compartment: str = 'cell_compartment'
    extracellular_value: int = 0
    cytoplasmic_value: int = 1
    nucleus_value: int = 2

@dataclass
class StandardBoundaryFields:
    filename: str = 'boundaries.parquet'
    id: str = 'cell_id'
    boundary_type: str = 'boundary_type'
    cell_value: str = 'cell'
    nucleus_value: str = 'nucleus'
    contains_nucleus: str = 'contains_nucleus'


# TODO: Add description
@dataclass
class TrainingTranscriptFields(StandardTranscriptFields):
    cell_encoding: str = 'cell_encoding'
    gene_encoding: str = 'gene_encoding'
    cell_cluster: str = 'cell_cluster'
    gene_cluster: str = 'gene_cluster'

@dataclass
class TrainingBoundaryFields(StandardBoundaryFields):
    index: str = 'entity_index'
    cell_encoding: str = 'cell_encoding'
    cell_cluster: str = 'cell_cluster'
