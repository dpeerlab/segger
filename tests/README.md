# Segger v0.2.0 Tests

## Quick Start

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests (requires full GPU dependencies)
PYTHONPATH=src pytest tests/ -v

# Run specific test modules
PYTHONPATH=src pytest tests/test_alignment_loss.py -v
PYTHONPATH=src pytest tests/test_fragment_mode.py -v
```

## Test Categories

### Unit Tests (CPU-only)
These tests can run without GPU dependencies:

```bash
# Fragment mode (uses scipy fallback)
PYTHONPATH=src pytest tests/test_fragment_mode.py -v

# Quality filters
PYTHONPATH=src pytest tests/test_quality_filter.py -v

# Field definitions
PYTHONPATH=src pytest tests/test_fields.py -v

# Optional dependency helpers
PYTHONPATH=src pytest tests/test_optional_deps.py -v

# Sample outputs (CPU-only helper)
PYTHONPATH=src pytest tests/test_sample_outputs.py -v

# SpatialData I/O
PYTHONPATH=src pytest tests/test_spatialdata_io.py -v

# Merged writer
PYTHONPATH=src pytest tests/test_merged_writer.py -v
```

### GPU-Required Tests
These tests require torch_scatter, cupy, cugraph:

```bash
# Alignment loss (requires torch)
PYTHONPATH=src pytest tests/test_alignment_loss.py -v

# Prediction graph (requires cupy/cugraph)
PYTHONPATH=src pytest tests/test_prediction_graph.py -v
```

## Dependencies by Test Module

| Test Module | Required Packages |
|-------------|-------------------|
| test_alignment_loss.py | torch, numpy |
| test_alignment_loss_integration.py | torch, anndata, scanpy, scipy |
| test_fragment_mode.py | numpy, polars, scipy |
| test_prediction_graph.py | torch, geopandas, shapely, cupy* |
| test_xenium_export.py | numpy, pandas, polars, zarr |
| test_quality_filter.py | polars |
| test_spatialdata_io.py | polars, geopandas, zarr |
| test_merged_writer.py | polars |

*GPU packages have CPU fallbacks

## Running Without GPU

For CI/CD or machines without GPU:

```bash
# Set environment variable to force CPU mode
export SEGBENCH_NO_GPU=1

# Run CPU-compatible tests only
PYTHONPATH=src pytest tests/test_fragment_mode.py tests/test_quality_filter.py tests/test_spatialdata_io.py -v
```

## Test Fixtures

Common fixtures are defined in `conftest.py`:

- `toy_transcripts` - Synthetic Xenium transcripts
- `toy_cells` - Synthetic cell metadata
- `toy_boundaries` - Synthetic cell boundaries (GeoDataFrame)
- `tmp_output_dir` - Temporary directory for test outputs

## Adding New Tests

1. Create test file: `tests/test_<feature>.py`
2. Add docstring with requirements and run command
3. Use fixtures from `conftest.py` where possible
4. Mark GPU-required tests with `@pytest.mark.gpu`
